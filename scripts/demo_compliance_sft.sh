#!/bin/bash

# This is a demo of verl doing PPO on GSM8k with a Qwen2.5-0.5B model

set -x
export WANDB_ENTITY=guardian-models
num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
python scripts/download_data.py \
  --dataset tomg-group-umd/compliance \
  --subset compliance \
  --split train_32000_mix \
  --num_examples -1 \
  --redownload \
  --local_dir data/compliance

PYTHONUNBUFFERED=1 \
    torchrun --nproc_per_node=$num_gpus \
    -m verl.trainer.fsdp_sft_trainer \
    model.partial_pretrain=Qwen/Qwen3-0.6B \
    data.train_files=data/compliance/train.parquet \
    data.val_files=data/compliance/val.parquet \
    data.prompt_key=verl_stuff \
    data.response_key=verl_stuff \
    data.prompt_dict_keys=["question"] \
    data.response_dict_keys=["answer"] \
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=32 \
    optim.lr=1e-4 \
    trainer.total_epochs=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=sft-gsm8k \
    trainer.experiment_name=verl_demo \
2>&1 | tee verl_demo.log
