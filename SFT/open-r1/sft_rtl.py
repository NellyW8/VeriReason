import logging
import os
import sys
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import SFTConfig
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

logger = logging.getLogger(__name__)


@dataclass
class RTLCoderScriptArguments(ScriptArguments):
    """
    Script arguments specific to RTL-Coder dataset fine-tuning.
    """
    dataset_files: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the JSON dataset file for RTL-Coder"}
    )


@dataclass
class RTLModelConfig(ModelConfig):
    """
    Model configuration with additional PEFT parameters.
    """
    peft_type: Optional[str] = field(
        default=None,
        metadata={"help": "Type of parameter-efficient fine-tuning to use"}
    )


def preprocess_rtl_dataset(examples, tokenizer):
    """
    Preprocess RTL-Coder dataset by formatting instructions and responses.
    
    The RTL-Coder dataset contains entries with "Instruction" and "Response" fields,
    where "Response" is a list of RTL code snippets.
    """
    texts = []
    
    print("\n" + "="*80)
    print("PREPROCESSING RTL DATASET")
    print("="*80)
    print(f"Processing {len(examples['Instruction'])} examples")
    
    for i, (instruction, responses) in enumerate(zip(examples["Instruction"], examples["Response"])):
        try:
            # Handle different possible formats of the Response field
            if isinstance(responses, list):
                if len(responses) > 0:
                    response = responses[0]  # Take the first response
                else:
                    response = "No response provided."
            elif isinstance(responses, str):
                response = responses
            elif isinstance(responses, dict):
                # Some datasets might have structured responses
                response = json.dumps(responses, indent=2)
            else:
                response = str(responses)
            
            # Ensure both instruction and response are strings
            instruction = str(instruction) if instruction is not None else ""
            
            # Print detailed info for first 3 examples
            if i < 3:
                print(f"\nEXAMPLE #{i+1}")
                print(f"INSTRUCTION: {instruction[:150]}..." if len(instruction) > 150 else f"INSTRUCTION: {instruction}")
                print(f"RESPONSE: {response[:150]}..." if len(response) > 150 else f"RESPONSE: {response}")
            
            # Format according to Qwen chat template format
            if "Qwen" in tokenizer.name_or_path:
                formatted_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            else:
                # Generic format for other models
                formatted_text = f"<|user|>\n{instruction}\n<|assistant|>\n{response}"
            
            if i < 3:
                print(f"FORMATTED TEXT:")
                print(formatted_text[:200] + "..." if len(formatted_text) > 200 else formatted_text)
            
            texts.append(formatted_text)
        except Exception as e:
            logger.warning(f"Error processing example: {e}")
            # Add a placeholder to maintain batch size
            texts.append("<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<|im_end|>")
    
    # Print sample of final formatted texts
    if len(texts) > 0:
        print("\nFINAL FORMATTED TEXT SAMPLE (what will be tokenized):")
        print(texts[0][:200] + "..." if len(texts[0]) > 200 else texts[0])
    print("="*80 + "\n")
    
    return {"text": texts}


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
    if hasattr(script_args, 'dataset_files') and script_args.dataset_files:
        # Load from local JSON file
        logger.info(f"Loading dataset from local file: {script_args.dataset_files}")
        
    try:
        with open(script_args.dataset_files, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Print dataset structure
        print("\n" + "="*80)
        print("DATASET STRUCTURE")
        print("="*80)
        print(f"Type of loaded data: {type(data)}")
        if isinstance(data, list) and len(data) > 0:
            print(f"Number of examples: {len(data)}")
            print(f"Example keys in first item: {list(data[0].keys())}")
            print(f"Sample first item: {json.dumps(data[0], indent=2)[:500]}...")
        elif isinstance(data, dict):
            print(f"Top-level keys: {list(data.keys())}")
        print("="*80 + "\n")
        
        # Check if data is a list (as expected)
        if not isinstance(data, list):
            logger.warning(f"Expected data to be a list, but got {type(data)}. Attempting to convert...")
            # If data is a dictionary, check if it has a key that might contain our list
            if isinstance(data, dict):
                possible_list_keys = ['data', 'instances', 'examples', 'samples', 'items']
                found_key = None
                for key in possible_list_keys:
                    if key in data and isinstance(data[key], list):
                        logger.info(f"Found list data under key '{key}'")
                        data = data[key]
                        found_key = key
                        break
                
                if found_key is None:
                    # If no list found, but there's only one key, try that
                    if len(data) == 1:
                        only_key = list(data.keys())[0]
                        if isinstance(data[only_key], list):
                            logger.info(f"Found list data under single key '{only_key}'")
                            data = data[only_key]
                        else:
                            logger.error(f"No suitable list data found in JSON. Top level keys: {list(data.keys())}")
                            sys.exit(1)
                    else:
                        logger.error(f"Expected a list of examples, but got a dictionary with keys: {list(data.keys())}")
                        sys.exit(1)
        
        # Now safely check the first few items
        sample_count = min(5, len(data))
        sample_items = data[:sample_count]
        
        if not all(["Instruction" in item and "Response" in item for item in sample_items]):
            logger.warning("Dataset JSON structure may not be as expected. Checking first few entries...")
            for i, item in enumerate(sample_items):
                logger.warning(f"Entry {i} keys: {item.keys()}")
        
        # Convert to Dataset format
        rtl_dataset = datasets.Dataset.from_dict({
            "Instruction": [item.get("Instruction", "") for item in data],
            "Response": [item.get("Response", ["No response"]) for item in data],
        })
        
        # Print dataset info after processing
        print("\n" + "="*80)
        print("CONVERTED DATASET INFO")
        print("="*80)
        print(f"Dataset size: {len(rtl_dataset)}")
        print(f"Dataset features: {rtl_dataset.features}")
        print(f"Sample row: {rtl_dataset[0]}")
        print("="*80 + "\n")
        
        rtl_dataset = rtl_dataset.select(range(0, len(rtl_dataset), 4))
        # Split dataset (80% train, 20% test)
        dataset = rtl_dataset.train_test_split(test_size=0.2, seed=training_args.seed)
        
        # Print split info
        print(f"Train split size: {len(dataset['train'])}")
        print(f"Test split size: {len(dataset['test'])}")
        
        # Set default splits
        if not hasattr(script_args, 'dataset_train_split') or script_args.dataset_train_split not in dataset:
            logger.info(f"Setting train split to 'train'")
            script_args.dataset_train_split = "train"
        if not hasattr(script_args, 'dataset_test_split') or script_args.dataset_test_split not in dataset:
            logger.info(f"Setting test split to 'test'")
            script_args.dataset_test_split = "test"
            
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file: {script_args.dataset_files}")
        logger.error(f"JSON error details: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error processing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loaded tokenizer: {tokenizer.__class__.__name__}")
    print(f"Model name or path: {tokenizer.name_or_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")

    # Preprocess dataset
    processed_dataset = {}
    for split in ["train", "test"]:
        if split in dataset:
            processed_dataset[split] = dataset[split].map(
                lambda examples: preprocess_rtl_dataset(examples, tokenizer),
                batched=True,
                remove_columns=dataset[split].column_names,
                desc=f"Preprocessing {split} dataset",
            )
    
    # Print processed dataset info
    print("\n" + "="*80)
    print("PROCESSED DATASET INFO")
    print("="*80)
    for split in processed_dataset:
        print(f"{split.upper()} split features: {processed_dataset[split].features}")
        print(f"{split.upper()} split size: {len(processed_dataset[split])}")
        if len(processed_dataset[split]) > 0:
            print(f"{split.upper()} sample: {processed_dataset[split][0]}")
    print("="*80 + "\n")

    # Examine tokenized sample
    if "train" in processed_dataset and len(processed_dataset["train"]) > 0:
        print("\n" + "="*80)
        print("EXAMINING TOKENIZED SAMPLE")
        print("="*80)
        
        sample_text = processed_dataset["train"][0]["text"]
        print(f"Raw sample text: {sample_text[:200]}...")
        
        # Tokenize the sample
        tokenized = tokenizer(sample_text, return_tensors="pt")
        print(f"Tokenized shape: {tokenized.input_ids.shape}")
        print(f"Token IDs (first 20): {tokenized.input_ids[0][:20].tolist()}")
        
        # Decode back to text to verify
        decoded = tokenizer.decode(tokenized.input_ids[0])
        print(f"Decoded sample: {decoded[:200]}...")
        print("="*80 + "\n")

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    ############################
    # Initialize the SFT Trainer
    ############################
    # Load model first
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    # Then initialize the trainer without the tokenizer parameter
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset.get(script_args.dataset_train_split),
        eval_dataset=processed_dataset.get(script_args.dataset_test_split) if training_args.eval_strategy != "no" else None,
        # tokenizer parameter removed
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    # Use our preprocessed text field
    )
    
    # Print SFT trainer info
    print("\n" + "="*80)
    print("SFT TRAINER INFO")
    print("="*80)
    print(f"Training with dataset size: {len(processed_dataset.get(script_args.dataset_train_split, []))}")
    if training_args.eval_strategy != "no":
        print(f"Evaluation with dataset size: {len(processed_dataset.get(script_args.dataset_test_split, []))}")
    print(f"Using PEFT: {get_peft_config(model_args) is not None}")
    print("="*80 + "\n")

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(processed_dataset.get(script_args.dataset_train_split, []))
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
        "dataset_name": script_args.dataset_files if hasattr(script_args, 'dataset_files') and script_args.dataset_files else script_args.dataset_name,
        "tags": ["open-r1", "rtl-coder", "verilog"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
        
        # Test generation with a simple example
        try:
            # Use first test example for generation test
            test_example = dataset[script_args.dataset_train_split][0]
            test_instruction = test_example["Instruction"]
            
            print("\n" + "="*80)
            print("GENERATION TEST")
            print("="*80)
            print(f"Test instruction: {test_instruction[:200]}...")
            
            # Format messages for generation
            if "Qwen" in tokenizer.name_or_path:
                messages = [
                    {"role": "user", "content": test_instruction}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Generic format
                prompt = f"<|user|>\n{test_instruction}\n<|assistant|>\n"
            
            print("\nGeneration prompt:")
            print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
            
            # Generate response
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            print(f"Input token count: {input_ids.shape[1]}")
            
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
            )
            
            # Decode full output
            full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print("\nFull generated output:")
            print(full_output[:300] + "..." if len(full_output) > 300 else full_output)
            
            # Trim prompt if needed
            generated_text = full_output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
                print("\nTrimmed response:")
                print(generated_text[:300] + "..." if len(generated_text) > 300 else generated_text)
            
            # Save example
            example_path = os.path.join(training_args.output_dir, "generation_example.json")
            with open(example_path, "w") as f:
                json.dump({
                    "instruction": test_instruction,
                    "prompt": prompt,
                    "full_output": full_output,
                    "trimmed_response": generated_text
                }, f, indent=2)
            
            print(f"\nGeneration example saved to: {example_path}")
            print("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"Error during generation test: {e}")
            import traceback
            traceback.print_exc()

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(processed_dataset.get(script_args.dataset_test_split, []))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    try:
        # Use our extended argument classes with the TrlParser
        parser = TrlParser((RTLCoderScriptArguments, SFTConfig, RTLModelConfig))
        script_args, training_args, model_args = parser.parse_args_and_config()
        
        # Log key parameters for debugging
        print("Starting SFT training with parameters:")
        print(f"  Model: {model_args.model_name_or_path}")
        print(f"  Dataset file: {script_args.dataset_files if hasattr(script_args, 'dataset_files') else 'None'}")
        print(f"  Output dir: {training_args.output_dir}")
        
        main(script_args, training_args, model_args)
    except Exception as e:
        logger.error(f"Error during script execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)