#!/bin/bash
# =============================================================================
# RTL-Coder Fine-tuning Launch Script
# - Distributed training on 4 GPUs using DeepSpeed ZeRO-3
# - LoRA fine-tuning for Qwen2.5-Coder-7B model
# =============================================================================

# Environment Configuration
export CUDA_VISIBLE_DEVICES=1,2,3,4

# Launch Training with Accelerate
accelerate launch \
  --config_file=recipes/accelerate_configs/zero3.yaml \
  --main_process_port=29557 \
  src/open_r1/sft_rtl.py \
  \
  # Model Configuration
  --model_name_or_path Qwen/Qwen2.5-Coder-7B \
  --dataset_name rtlcoder \
  --dataset_files PATH_TO_DATASET \
  \
  # Training Parameters
  --learning_rate 3.0e-5 \
  --num_train_epochs 1 \
  --max_seq_length 4096 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 5 \
  --eval_strategy steps \
  --eval_steps 100 \
  \
  # Optimization Settings
  --gradient_checkpointing \
  --bf16 \
  --packing \
  \
  # LoRA Parameters
  --peft_type lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  \
  # Output Configuration
  --output_dir data/Qwen2.5-Coder-7B-LoRA