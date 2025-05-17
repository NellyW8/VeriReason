import logging
import os
import sys
import re
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from open_r1.verilog_rewards_tb import verilog_reward, verilog_format_reward, extract_verilog_code
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO Verilog training script.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["verilog", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'verilog', 'format'"
        },
    )
    dataset_prompt_column: str = field(
        default="instruction",
        metadata={"help": "Column name containing the problem statement"}
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Dataset split for training"}
    )
    dataset_test_split: str = field(
        default="test",
        metadata={"help": "Dataset split for evaluation"}
    )
    # Add parameters for dataset loading
    dataset_files: str = field(
        default=None,
        metadata={"help": "Path to the dataset JSON file"}
    )
    dataset_name: str = field(
        default=None, 
        metadata={"help": "Hugging Face Hub dataset name, e.g., 'Nellyw888/RTL-Coder_7b_reasoning'"}
    )
    dataset_revision: str = field(
        default=None,
        metadata={"help": "Revision/version of the dataset to load from HF Hub"}
    )
    dataset_sample_size: int = field(
        default=None,
        metadata={"help": "Number of samples to use from the dataset (None for all)"}
    )


def extract_module_name(instruction):
    """Extract module name from instruction."""
    # Try to find a module name in the instruction
    pattern = r"module\s+(\w+)|implement\s+(?:a|the)\s+(\w+)\s+module|design\s+(?:a|the)\s+(\w+)"
    matches = re.findall(pattern, instruction, re.IGNORECASE)
    
    if matches:
        # Flatten the matches and take the first non-empty group
        for match in matches:
            if isinstance(match, tuple):
                for group in match:
                    if group:
                        return group
            elif match:
                return match
    
    return None


def clean_verilog_output(output):
    """Extract clean Verilog code from formatted output."""
    # First try to extract just the Verilog code from the answer tags
    verilog_match = re.search(r'<answer>\s*```verilog\s*([\s\S]*?)\s*```\s*</answer>', output)
    if verilog_match:
        return verilog_match.group(1).strip()
    
    # If no verilog tag, try just code block in answer tags
    match = re.search(r'<answer>\s*```\s*([\s\S]*?)\s*```\s*</answer>', output)
    if match:
        return match.group(1).strip()
    
    # If we can't find the expected format, just extract any code block
    code_block_match = re.search(r'```(?:verilog|v)?\s*([\s\S]*?)\s*```', output)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # If all else fails, just return the full output
    return output


def preprocess_dataset_entry(entry):
    """
    Preprocess a single dataset entry to extract all necessary fields.
    
    Returns:
        dict: Processed entry with standardized fields
    """
    # Initialize the processed entry
    processed = {}
    
    # Get the instruction field
    if "instruction" in entry:
        processed["instruction"] = entry["instruction"]
    elif "Instruction" in entry:
        processed["instruction"] = entry["Instruction"]
    else:
        processed["instruction"] = ""
        logger.warning("No instruction field found in dataset entry")
    
    # Extract golden code from output field
    if "output" in entry:
        raw_output = entry["output"]
        processed["golden_code"] = clean_verilog_output(raw_output)
    elif "Response" in entry:
        raw_output = entry["Response"]
        processed["golden_code"] = clean_verilog_output(raw_output)
    else:
        processed["golden_code"] = ""
        logger.warning("No output/response field found in dataset entry")
    
    # Get testbench code
    if "tb" in entry:
        processed["testbench_code"] = entry["tb"]
    else:
        processed["testbench_code"] = ""
    
    # Get testbench results
    if "tb_result" in entry:
        processed["testbench_result"] = entry["tb_result"]
    else:
        processed["testbench_result"] = ""
    
    # Try to extract module name
    module_name = extract_module_name(processed["instruction"])
    
    # If not found in instruction, try from golden code
    if not module_name and processed["golden_code"]:
        pattern = re.compile(r"module\s+(\w+)")
        match = pattern.search(processed["golden_code"])
        if match:
            module_name = match.group(1)
    
    # Add default if still not found
    if not module_name:
        module_name = "verilog_module"
    
    processed["module_name"] = module_name
    
    # Extract ports if possible from golden code
    if processed["golden_code"]:
        processed["ports"] = extract_ports_from_code(processed["golden_code"])
    else:
        processed["ports"] = []
    
    return processed


def extract_ports_from_code(verilog_code):
    """Extract port list from Verilog code."""
    ports = []
    
    # Try to match port declaration in module header
    port_pattern = re.compile(r"module\s+\w+\s*\((.*?)\)", re.DOTALL)
    port_match = port_pattern.search(verilog_code)
    
    if port_match:
        # Get the full port declaration and split by commas
        port_text = port_match.group(1).strip()
        raw_ports = [p.strip() for p in port_text.split(',')]
        
        # Clean up port names
        for port in raw_ports:
            # Remove any comments
            port = re.sub(r'//.*$|/\*.*?\*/', '', port)
            
            # If port contains a direction (input/output/inout)
            if re.search(r'\b(input|output|inout)\b', port):
                # Extract just the port name after the direction and any width declaration
                name_match = re.search(r'\b(input|output|inout)\b\s+(?:reg|wire)?\s*(?:\[.*?\])?\s*(\w+)', port)
                if name_match:
                    ports.append(name_match.group(2))
            else:
                # Just a port name without direction
                name_match = re.search(r'\b(\w+)\b', port)
                if name_match:
                    ports.append(name_match.group(1))
    
    return ports


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load datasets
    ################
    train_test_dataset = None
        
    if hasattr(script_args, 'dataset_name') and script_args.dataset_name:
        # Load from Hugging Face
        logger.info(f"Loading dataset from Hugging Face: {script_args.dataset_name}")
        
        try:
            # Determine dataset configuration and revision if provided
            config = script_args.dataset_config if hasattr(script_args, 'dataset_config') else None
            revision = script_args.dataset_revision if hasattr(script_args, 'dataset_revision') else None
            
            # Load raw dataset
            raw_dataset = datasets.load_dataset(
                script_args.dataset_name,
                config,
                revision=revision
            )
            
            logger.info(f"Successfully loaded dataset with splits: {list(raw_dataset.keys())}")
            
            # Determine which split to use (default to 'train' if available)
            available_splits = list(raw_dataset.keys())
            main_split = None
            
            if 'train' in available_splits:
                main_split = 'train'
            elif len(available_splits) > 0:
                main_split = available_splits[0]
            
            if main_split is None:
                logger.error(f"No suitable split found in dataset {script_args.dataset_name}")
                sys.exit(1)
            
            logger.info(f"Using split '{main_split}' as source data")
            data = raw_dataset[main_split]
            
            # Check dataset columns
            logger.info(f"Dataset columns: {list(data.features.keys())}")
            
            # Process the dataset specifically for our case with known columns
            # ['id', 'instruction', 'output', 'tb', 'tb_result']
            if 'instruction' in data.features and 'output' in data.features:
                logger.info("Found required columns 'instruction' and 'output'")
                
                # Extract golden code from output
                responses = data['output']
                golden_codes = [clean_verilog_output(response) for response in responses]
                
                logger.info(f"Extracted golden code for {len(golden_codes)} examples")
                if golden_codes and len(golden_codes) > 0:
                    logger.info(f"Sample golden code (first example): {golden_codes[0][:200]}...")
                
                # Create the dataset with our desired format
                dataset_dict = {
                    "Instruction": data['instruction'],
                    "Response": data['output'],
                    "golden_code": golden_codes,
                    "tb": data['tb'],
                    "tb_result": data['tb_result']
                }
                
                sample_idx = 0
                sample_entry = {
                    "Instruction": data['instruction'][sample_idx],
                    "Response": data['output'][sample_idx],
                    "golden_code": golden_codes[sample_idx],
                    "tb": data['tb'][sample_idx],
                    "tb_result": data['tb_result'][sample_idx]
                }
                print("Sample dataset entry:")
                print(sample_entry)
                # Convert to expected format
                rtl_dataset = datasets.Dataset.from_dict(dataset_dict)
            else:
                logger.error("Required columns 'instruction' or 'output' not found in dataset")
                sys.exit(1)
            
            # Split dataset (80% train, 20% test)
            train_test_dataset = rtl_dataset.train_test_split(test_size=0.2, seed=training_args.seed)
            
            # Set default splits
            if not hasattr(script_args, 'dataset_train_split') or script_args.dataset_train_split not in train_test_dataset:
                logger.info(f"Setting train split to 'train'")
                script_args.dataset_train_split = "train"
            if not hasattr(script_args, 'dataset_test_split') or script_args.dataset_test_split not in train_test_dataset:
                logger.info(f"Setting test split to 'test'")
                script_args.dataset_test_split = "test"
                
        except Exception as e:
            logger.error(f"Error loading dataset from Hugging Face: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    if train_test_dataset is None:
        logger.error("No dataset loaded. Please provide a valid dataset name.")
        sys.exit(1)

    #print sample dataset
    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    def verilog_reward_wrapper(completions, **kwargs):
        # Extract parameters needed by the reward function
        golden_code = kwargs.get("golden_code", None)  # The clean Verilog code
        testbench_code = kwargs.get("tb", None)  # The testbench code
        testbench_result = kwargs.get("tb_result", None)  # The testbench results
        
        # Remove extracted parameters from kwargs to avoid passing them twice
        kwargs_copy = kwargs.copy()
        for key in ["golden_code", "tb", "tb_result"]:
            kwargs_copy.pop(key, None)
        
        # Call the verilog_reward function with the extracted data
        return verilog_reward(
            completions=completions,
            golden_code=golden_code,
            testbench_code=testbench_code,
            testbench_result=testbench_result,
            **kwargs_copy  # Use the modified kwargs that doesn't contain the extracted parameters
        )

    # Register reward functions
    REWARD_FUNCS_REGISTRY = {
        "verilog": verilog_reward_wrapper,
        "format": verilog_format_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example, prompt_column: str = "instruction"):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            # Check for alternative column names
            if "Instruction" in example:
                prompt.append({"role": "user", "content": example["Instruction"]})
            elif "instruction" in example:
                prompt.append({"role": "user", "content": example["instruction"]})
            else:
                raise ValueError(f"Dataset Question Field Error: no suitable instruction column found. Available columns: {example.keys()}")
        else:
            prompt.append({"role": "user", "content": example[prompt_column]})
            
        return {"prompt": prompt}

    # Apply the conversation formatting
    train_test_dataset = train_test_dataset.map(make_conversation)
    print("train_test_dataset loaded successfully!")
    # Remove old messages column if present
    for split in train_test_dataset:
        if "messages" in train_test_dataset[split].column_names:
            train_test_dataset[split] = train_test_dataset[split].remove_columns("messages")


    # Initialize model kwargs
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_test_dataset["train"],
        eval_dataset=train_test_dataset["test"] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )
    
    # Add these logging statements to your code after preprocessing
    logger.info(f"Total dataset size after preprocessing: {len(train_test_dataset['train']) + len(train_test_dataset['test'])}")
    logger.info(f"Training dataset size after split: {len(train_test_dataset['train'])}")
    logger.info(f"Test dataset size after split: {len(train_test_dataset['test'])}")
    
    # Log sample example to verify data is correct
    if len(train_test_dataset["train"]) > 0:
        sample = train_test_dataset["train"][0]
        logger.info("Sample training example:")
        logger.info(f"  Instruction: {sample['Instruction'][:100]}...")
        logger.info(f"  Has golden code: {'Yes' if sample.get('golden_code') else 'No'}")
        logger.info(f"  Has testbench: {'Yes' if sample.get('testbench_code') else 'No'}")
        logger.info(f"  Has testbench result: {'Yes' if sample.get('testbench_result') else 'No'}")
    
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    # checkpoint = None
    #if training_args.resume_from_checkpoint is not None:
        #checkpoint = training_args.resume_from_checkpoint
    #elif #last_checkpoint is not None:
       # c#heckpoint = last_checkpoint
    #train_result = trainer.train(resume_from_checkpoint=checkpoint)
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_test_dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": "RTL-Coder-GRPO",
        "tags": ["open-r1", "verilog", "rtl-coder"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(train_test_dataset["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)