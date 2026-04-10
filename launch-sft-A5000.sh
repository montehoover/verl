# launch launch-sft-A5000.sh --classical_logfile_names --gpu_type rtxa5000 --mem 128 --timelimit 72 --gpus 8

# python run_sft.py \
#     --model Qwen/Qwen3-8B \
#     --dataset montehoover/DynaBench \
#     --data_download_dir data/dynabench1 --subset DynaBenchSafetyMix --val_split validation \
#     --lr 1e-4 --batch_size 128 \
#     --lora_rank 16 --lora_alpha 8 --lora_target_modules all-linear \
#     --wandb_entity azheng15-umd \
#     --wandb_project DynaGuard2 \
#     --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_sft \
#     --resume_training --save_freq 10 --overwrite \
#     --max_length 8192 --truncation left

# python run_sft.py \
#     --model Qwen/Qwen3-8B \
#     --dataset montehoover/DynaBench \
#     --data_download_dir data/dynabench1 --subset DynaBenchSafetyMix --val_split validation \
#     --lr 1e-6 --batch_size 128 \
#     --lora_rank 16 --lora_alpha 8 --lora_target_modules all-linear \
#     --wandb_entity azheng15-umd \
#     --wandb_project DynaGuard2 \
#     --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_sft \
#     --resume_training --save_freq 10 --overwrite \
#     --max_length 8192 --truncation left

# python run_sft.py \
#     --model Qwen/Qwen3-8B \
#     --dataset montehoover/DynaBench \
#     --data_download_dir data/dynabench1 --subset DynaBenchSafetyMix --val_split validation \
#     --lr 7.5e-7 --batch_size 128 \
#     --lora_rank 16 --lora_alpha 8 --lora_target_modules all-linear \
#     --wandb_entity azheng15-umd \
#     --wandb_project DynaGuard2 \
#     --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
#     --resume_training --save_freq 10 --overwrite \
#     --max_length 8192 --truncation left

# python run_sft.py \
#     --model Qwen/Qwen3-8B \
#     --dataset montehoover/DynaBench \
#     --data_download_dir data/dynabench1 --subset DynaBenchSafetyMix --val_split validation \
#     --lr 5e-7 --batch_size 128 \
#     --lora_rank 16 --lora_alpha 8 --lora_target_modules all-linear \
#     --wandb_entity azheng15-umd \
#     --wandb_project DynaGuard2 \
#     --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
#     --resume_training --save_freq 10 --overwrite \
#     --max_length 8192 --truncation left

# python run_sft.py \
#     --model Qwen/Qwen3-8B \
#     --dataset montehoover/DynaBench \
#     --data_download_dir data/dynabench1 --subset DynaBenchSafetyMix --val_split validation \
#     --lr 2.5e-7 --batch_size 128 \
#     --lora_rank 16 --lora_alpha 8 --lora_target_modules all-linear \
#     --wandb_entity azheng15-umd \
#     --wandb_project DynaGuard2 \
#     --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
#     --resume_training --save_freq 10 --overwrite \
#     --max_length 8192 --truncation left

# python run_sft.py \
#     --model Qwen/Qwen3-8B \
#     --dataset montehoover/DynaBench \
#     --data_download_dir data/dynabench1 --subset DynaBenchSafetyMix --val_split validation \
#     --lr 1e-7 --batch_size 128 \
#     --lora_rank 16 --lora_alpha 8 --lora_target_modules all-linear \
#     --wandb_entity azheng15-umd \
#     --wandb_project DynaGuard2 \
#     --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
#     --resume_training --save_freq 10 --overwrite \
#     --max_length 8192 --truncation left

# python run_sft.py \
#     --model Qwen/Qwen3-8B \
#     --dataset montehoover/DynaBench \
#     --data_download_dir data/dynabench1 --subset DynaBenchSafetyMix --val_split validation \
#     --lr 1e-4 --batch_size 128 \
#     --lora_rank 16 --lora_alpha 8 --lora_target_modules all-linear \
#     --wandb_entity azheng15-umd \
#     --wandb_project DynaGuard2 \
#     --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
#     --resume_training --save_freq 10 --overwrite \
#     --max_length 8192 --truncation left


### SANITY CHECK, run smaller number examples
python run_sft.py \
    --model Qwen/Qwen3-8B \
    --dataset montehoover/DynaBench \
    --data_download_dir data/dynabench1 --subset DynaBenchSafetyMix --val_split validation \
    --lr 1e-4 --lr_schedule wsd --batch_size 128 --num_examples 2000 --epochs 4 \
    --lora_rank 16 --lora_alpha 8 --lora_target_modules all-linear \
    --wandb_entity azheng15-umd \
    --wandb_project DynaGuard2 \
    --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
    --resume_training --save_freq 10 --overwrite \
    --max_length 8192 --truncation left

# python run_sft.py \
#     --model Qwen/Qwen3-8B \
#     --dataset montehoover/DynaBench \
#     --data_download_dir data/dynabench1 --subset DynaBenchSafetyMix --val_split validation \
#     --lr 1e-5 --lr_schedule wsd --batch_size 128 --num_examples 2000 --epochs 4 \
#     --lora_rank 16 --lora_alpha 8 --lora_target_modules all-linear \
#     --wandb_entity azheng15-umd \
#     --wandb_project DynaGuard2 \
#     --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
#     --resume_training --save_freq 10 --overwrite \
#     --max_length 8192 --truncation left

# python run_sft.py \
#     --model Qwen/Qwen3-8B \
#     --dataset montehoover/DynaBench \
#     --data_download_dir data/dynabench1 --subset DynaBenchSafetyMix --val_split validation \
#     --lr 1e-4 --lr_schedule cosine --batch_size 128 --num_examples 2000 --epochs 4 \
#     --lora_rank 16 --lora_alpha 8 --lora_target_modules all-linear \
#     --wandb_entity azheng15-umd \
#     --wandb_project DynaGuard2 \
#     --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
#     --resume_training --save_freq 10 --overwrite \
#     --max_length 8192 --truncation left

# python run_sft.py \
#     --model Qwen/Qwen3-8B \
#     --dataset montehoover/DynaBench \
#     --data_download_dir data/dynabench1 --subset DynaBenchSafetyMix --val_split validation \
#     --lr 1e-5 --lr_schedule cosine --batch_size 128 --num_examples 2000 --epochs 4 \
#     --lora_rank 16 --lora_alpha 8 --lora_target_modules all-linear \
#     --wandb_entity azheng15-umd \
#     --wandb_project DynaGuard2 \
#     --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
#     --resume_training --save_freq 10 --overwrite \
#     --max_length 8192 --truncation left