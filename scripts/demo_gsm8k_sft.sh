#!/bin/bash

# This is a demo of verl doing PPO on GSM8k with a Qwen2.5-0.5B model

set -x
export WANDB_ENTITY=guardian-models
num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
python examples/data_preprocess/gsm8k.py --local_dir data/gsm8k

PYTHONUNBUFFERED=1 \
    torchrun --nproc_per_node=$num_gpus \
    -m verl.trainer.fsdp_sft_trainer \
    model.partial_pretrain=Qwen/Qwen3-0.6B \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=8 \
    data.train_batch_size=256 \
    optim.lr=1e-5 \
    trainer.total_epochs=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=sft-gsm8k \
    trainer.experiment_name=verl_demo \
2>&1 | tee verl_demo.log
