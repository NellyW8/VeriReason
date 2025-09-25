# VeriReason Repository

This repository contains tools and configurations for training language models for the paper: 


## Project Description

This study introduces VeriReason, a novel approach utilizing reinforcement learning with testbench feedback to enhance the performance of pre-trained models for Verilog RTL code generation. VeriReason-Qwen2.5-3B is a 3B parameter model based on Qwen2.5-Coder-3B that combines supervised fine-tuning with Guided Reward Proximal Optimization (GRPO) reinforcement learning, specifically tailored for RTL code generation.
The model integrates explicit reasoning capabilities with reinforcement learning for Verilog generation, establishing a new state-of-the-art for automated RTL synthesis in a smaller model size. By using our curated high-quality training examples alongside a feedback-driven reward model, this 3B parameter model delivers exceptional performance on Verilog generation tasks while maintaining efficiency.

<p align="center">
  <img src="assets/Verireason_workflow.png" alt="VeriReason Workflow" width="800"/>
</p>

## Training Options

### Supervised Fine-Tuning (SFT)

You can use either of the following methods to train an SFT model:

#### Using LLamaFactory

```bash
llamafactory-cli train qwen2.5_7b.yaml
```

#### Using OpenR1

1. Move `sft_rtl` to the folder: `src/open_r1/`
2. Make the training script executable:
   ```bash
   chmod +x run_rtl_training.sh
   ```
3. Run the training script:
   ```bash
   ./run_rtl_training.sh
   ```

### GRPO Training

For GRPO (Generative Reinforcement Learning from Preference Optimization) training:

1. Move the necessary files to the OpenR1 directory:
   ```bash
   mv verilog_rewards_tb.py verilog_train_tb.py src/open-r1/
   ```
   
2. Create a new directory for the Verilog recipe:
   ```bash
   mkdir verilog_recipe
   mv verilog_grpo_tb.yaml verilog_recipe/
   ```

3. Example training command:
   ```bash
   NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=5,6,7 ACCELERATE_USE_NCCL=1 accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=3 src/open_r1/verilog_train_rtlcoder.py --config verilog_recipe/verilog_grpo_tb.yaml --use_vllm=false
   ```

## Datasets

The following datasets are available on Hugging Face:

| Dataset | Description | Link |
|---------|-------------|------|
| RTL-Coder_small | Filtered dataset with no reasoning | [Link](https://huggingface.co/datasets/Nellyw888/RTL-Coder_small) |
| RTL-Coder_7b_reasoning_tb_simple | VeriReason simple dataset with reasoning and testbench | [Link](https://huggingface.co/datasets/Nellyw888/RTL-Coder_7b_reasoning_tb_simple) |
| RTL-Coder_7b_reasoning_tb | VeriReason hard dataset with reasoning and testbench | [Link](https://huggingface.co/datasets/Nellyw888/RTL-Coder_7b_reasoning_tb) |
| RTL-Coder_7b_reasoning_tb_combined | VeriReason combined dataset with reasoning and testbench | [Link](https://huggingface.co/datasets/Nellyw888/RTL-Coder_7b_reasoning_tb_combined) |

## Model checkpoints

The following fine-tuned models are available on Hugging Face:

| Model | Description | Link |
|-------|-------------|------|
| VeriReason-Qwen2.5-1.5B | 1.5B parameter model based on Qwen2.5 | [Link](https://huggingface.co/Nellyw888/VeriReason-Qwen2.5-1.5B-grpo-small) |
| VeriReason-Qwen2.5-3B | 3B parameter model based on Qwen2.5 with RTL GRPO | [Link](https://huggingface.co/Nellyw888/VeriReason-Qwen2.5-3B-Verilog-RTL-GRPO-reasoning-tb/settings) |
| VeriReason-Qwen2.5-7b | 7B parameter model based on Qwen2.5 with SFT Reasoning | [Link](https://huggingface.co/Nellyw888/VeriReason-Qwen2.5-7b-SFT-Reasoning) |
| VeriReason-Llama-7b | 7B parameter model based on Code Llama | [Link](https://huggingface.co/Nellyw888/VeriReason-Llama-7b-RTLCoder-GRPO-reasoning-tb) |

## Requirements

- CUDA-compatible GPUs
- PyTorch with CUDA support
- Accelerate library
- NCCL for distributed training


