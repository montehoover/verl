#!/bin/bash

MODEL=Qwen/Qwen3-0.6B
DATASET=tomg-group-umd/compliance
SUBSET=compliance
SPLIT=train_cot
EPOCHS=1
LR=1e-5
BATCH_SIZE=32
BATCH_SIZE_PER_GPU=2
EXPERIMENT_NAME=$(echo $MODEL | sed 's/.*\///')-${SPLIT}-lr${LR}-bs${BATCH_SIZE}

set -x
export WANDB_ENTITY=guardian-models
num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
python scripts/download_data.py \
  --dataset $DATASET \
  --subset $SUBSET \
  --split $SPLIT \
  --num_examples 1000 \
  --local_dir data/compliance \
  --redownload

PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=$num_gpus \
  -m verl.trainer.fsdp_sft_trainer \
  model.partial_pretrain=$MODEL \
  model.enable_gradient_checkpointing=True \
  data.train_files=data/compliance/train.parquet \
  data.val_files=data/compliance/val.parquet \
  data.prompt_key=verl_stuff \
  data.response_key=verl_stuff \
  data.prompt_dict_keys=["question"] \
  data.response_dict_keys=["answer"] \
  +data.add_system_prompt=True \
  data.max_length=8192 \
  data.micro_batch_size_per_gpu=$BATCH_SIZE_PER_GPU \
  data.train_batch_size=$BATCH_SIZE \
  optim.lr=$LR \
  trainer.total_epochs=$EPOCHS \
  trainer.logger=['console','wandb'] \
  trainer.project_name=sft-compliance \
  trainer.experiment_name=$EXPERIMENT_NAME \
2>&1 | tee log/$EXPERIMENT_NAME.log
