#!/bin/bash

# --- Default Values ---
MODEL="Qwen/Qwen3-0.6B"
DATASET="tomg-group-umd/compliance"
SUBSET="compliance"
SPLIT="train_cot"
EPOCHS=1
LR="1e-5"
BATCH_SIZE=32
BATCH_SIZE_PER_GPU=2
WANDB_ENTITY="guardian-models"
WANDB_PROJECT_NAME="sft-compliance"
NUM_EXAMPLES=1000
DOWNLOAD_LOCAL_DIR="data/compliance" # This will also be used for train/val files
REDOWNLOAD_ENABLED=false # Becomes true if --redownload is passed

# --- Help Message Function ---
usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "This script runs a PPO training process with configurable parameters."
  echo ""
  echo "Options:"
  echo "  --model MODEL_NAME             Model name (default: $MODEL)"
  echo "  --dataset DATASET_NAME         Dataset name (default: $DATASET)"
  echo "  --subset SUBSET_NAME           Dataset subset (default: $SUBSET)"
  echo "  --split SPLIT_NAME             Dataset split (default: $SPLIT)"
  echo "  --epochs NUM_EPOCHS            Number of epochs (default: $EPOCHS)"
  echo "  --lr LEARNING_RATE             Learning rate (default: $LR)"
  echo "  --batch_size BATCH_SIZE        Total batch size (default: $BATCH_SIZE)"
  echo "  --batch_size_per_gpu BS_GPU    Batch size per GPU (default: $BATCH_SIZE_PER_GPU)"
  echo "  --wandb_entity ENTITY          Weights & Biases entity (default: $WANDB_ENTITY)"
  echo "  --num_examples NUM             Number of examples to train on. -1 for all (default: $NUM_EXAMPLES)"
  echo "  --download_local_dir DIR       Local directory for data (default: $DOWNLOAD_LOCAL_DIR)"
  echo "  --redownload                   Force redownload of data (default: disabled)"
  echo "  --wandb_project_name NAME     Trainer project name for WandB (default: $WANDB_PROJECT_NAME)"
  echo "  -h, --help                     Display this help message and exit"
  exit 0
}

# --- Parse Command-Line Arguments ---
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -h|--help)
      usage
      ;;
    --model)
      if [[ -n "$2" && "${2:0:1}" != "-" ]]; then MODEL="$2"; shift 2; else echo "Error: --model requires a value." >&2; exit 1; fi ;;
    --dataset)
      if [[ -n "$2" && "${2:0:1}" != "-" ]]; then DATASET="$2"; shift 2; else echo "Error: --dataset requires a value." >&2; exit 1; fi ;;
    --subset)
      if [[ -n "$2" && "${2:0:1}" != "-" ]]; then SUBSET="$2"; shift 2; else echo "Error: --subset requires a value." >&2; exit 1; fi ;;
    --split)
      if [[ -n "$2" && "${2:0:1}" != "-" ]]; then SPLIT="$2"; shift 2; else echo "Error: --split requires a value." >&2; exit 1; fi ;;
    --epochs)
      if [[ -n "$2" && "${2:0:1}" != "-" ]]; then EPOCHS="$2"; shift 2; else echo "Error: --epochs requires a value." >&2; exit 1; fi ;;
    --lr)
      if [[ -n "$2" && "${2:0:1}" != "-" ]]; then LR="$2"; shift 2; else echo "Error: --lr requires a value." >&2; exit 1; fi ;;
    --batch_size)
      if [[ -n "$2" && "${2:0:1}" != "-" ]]; then BATCH_SIZE="$2"; shift 2; else echo "Error: --batch_size requires a value." >&2; exit 1; fi ;;
    --batch_size_per_gpu)
      if [[ -n "$2" && "${2:0:1}" != "-" ]]; then BATCH_SIZE_PER_GPU="$2"; shift 2; else echo "Error: --batch_size_per_gpu requires a value." >&2; exit 1; fi ;;
    --wandb_entity)
      if [[ -n "$2" && "${2:0:1}" != "-" ]]; then WANDB_ENTITY="$2"; shift 2; else echo "Error: --wandb_entity requires a value." >&2; exit 1; fi ;;
    --num_examples)
      if [[ -n "$2" && "${2:0:1}" != "-" ]]; then NUM_EXAMPLES="$2"; shift 2; else echo "Error: --num_examples requires a value." >&2; exit 1; fi ;;
    --download_local_dir)
      if [[ -n "$2" && "${2:0:1}" != "-" ]]; then DOWNLOAD_LOCAL_DIR="$2"; shift 2; else echo "Error: --download_local_dir requires a value." >&2; exit 1; fi ;;
    --redownload)
      REDOWNLOAD_ENABLED=true; shift ;; # Flag, no value needed
    --wandb_project_name)
      if [[ -n "$2" && "${2:0:1}" != "-" ]]; then WANDB_PROJECT_NAME="$2"; shift 2; else echo "Error: --wandb_project_name requires a value." >&2; exit 1; fi ;;
    *)
      echo "Error: Unknown option: $1" >&2
      usage # It's good practice to show usage on unknown option
      exit 1
      ;;
  esac
done

# --- Final Argument Setup ---
export WANDB_ENTITY # Export to environment for WandB
export PYTHONUNBUFFERED=1 # Ensure Python output is not buffered
EXPERIMENT_NAME=$(echo "$MODEL" | sed 's|.*/||')-${SPLIT}-lr${LR}-bs${BATCH_SIZE}
LOG_FILE="log/$EXPERIMENT_NAME.log"
TRAIN_FILES="$DOWNLOAD_LOCAL_DIR/train.parquet"
VAL_FILES="$DOWNLOAD_LOCAL_DIR/val.parquet"
num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")


################################
# Dataset
################################
DOWNLOAD_CMD_ARGS=(
  "--dataset" "$DATASET"
  "--subset" "$SUBSET"
  "--split" "$SPLIT"
  "--num_examples" "$NUM_EXAMPLES"
  "--local_dir" "$DOWNLOAD_LOCAL_DIR"
)
if [ "$REDOWNLOAD_ENABLED" = true ]; then
  DOWNLOAD_CMD_ARGS+=("--redownload")
fi
echo "Running download script..."
python scripts/download_data.py "${DOWNLOAD_CMD_ARGS[@]}"

################################
# SFT Training
################################
echo "Starting training..."
# The main torchrun command
torchrun --nproc_per_node="$num_gpus" \
  -m verl.trainer.fsdp_sft_trainer \
  model.partial_pretrain="$MODEL" \
  model.enable_gradient_checkpointing=True \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.prompt_key=verl_stuff \
  data.response_key=verl_stuff \
  data.prompt_dict_keys='["question"]' \
  data.response_dict_keys='["answer"]' \
  +data.add_system_prompt=True \
  data.max_length=8192 \
  data.micro_batch_size_per_gpu="$BATCH_SIZE_PER_GPU" \
  data.train_batch_size="$BATCH_SIZE" \
  optim.lr="$LR" \
  trainer.total_epochs="$EPOCHS" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="$WANDB_PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
2>&1 | tee "$LOG_FILE"

echo "Script finished. Log saved to $LOG_FILE"