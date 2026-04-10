# launch launch-grpo-A6000.sh --classical_logfile_names --gpu_type rtxa6000 --mem 123 --timelimit 72 --gpus 4

python run_grpo.py \
    --model Qwen/Qwen3-8B \
    --dataset ahans1/code_patrol_postprocessed \
    --data_download_dir data/code_patrol \
    --subset v2 --lr 1e-6 --batch_size 4 --rollout_batch_size 16 --vllm_model_shards 2 \
    --vllm_cache_utilization 0.6 \
    --num_generations 3 --max_prompt_length 8192 --max_response_length 1024 \
    --no-offload_weights_and_states \
    --wandb_entity azheng15-umd \
    --wandb_project DynaGuard2 \
    --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
    --resume_training \
    --overwrite

python run_grpo.py \
    --model Qwen/Qwen3-8B \
    --dataset ahans1/code_patrol_postprocessed \
    --data_download_dir data/code_patrol \
    --subset v3 --lr 1e-6 --batch_size 4 --rollout_batch_size 16 --vllm_model_shards 2 \
    --vllm_cache_utilization 0.6 \
    --num_generations 3 --max_prompt_length 8192 --max_response_length 1024 \
    --no-offload_weights_and_states \
    --wandb_entity azheng15-umd \
    --wandb_project DynaGuard2 \
    --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
    --resume_training \
    --overwrite

python run_grpo.py \
    --model Qwen/Qwen3-8B \
    --dataset ahans1/code_patrol_postprocessed \
    --data_download_dir data/code_patrol \
    --subset v4 --lr 1e-6 --batch_size 4 --rollout_batch_size 16 --vllm_model_shards 2 \
    --vllm_cache_utilization 0.6 \
    --num_generations 3 --max_prompt_length 8192 --max_response_length 1024 \
    --no-offload_weights_and_states \
    --wandb_entity azheng15-umd \
    --wandb_project DynaGuard2 \
    --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
    --resume_training \
    --overwrite

python run_grpo.py \
    --model Qwen/Qwen3-8B \
    --dataset ahans1/code_patrol_postprocessed \
    --data_download_dir data/code_patrol \
    --subset v5 --lr 1e-6 --batch_size 4 --rollout_batch_size 16 --vllm_model_shards 2 \
    --vllm_cache_utilization 0.6 \
    --num_generations 3 --max_prompt_length 8192 --max_response_length 1024 \
    --no-offload_weights_and_states \
    --wandb_entity azheng15-umd \
    --wandb_project DynaGuard2 \
    --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
    --resume_training \
    --overwrite

python run_grpo.py \
    --model Qwen/Qwen3-8B \
    --dataset ahans1/code_patrol_postprocessed \
    --data_download_dir data/code_patrol \
    --subset v6 --lr 1e-6 --batch_size 4 --rollout_batch_size 16 --vllm_model_shards 2 \
    --vllm_cache_utilization 0.6 \
    --num_generations 3 --max_prompt_length 8192 --max_response_length 1024 \
    --no-offload_weights_and_states \
    --wandb_entity azheng15-umd \
    --wandb_project DynaGuard2 \
    --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
    --resume_training \
    --overwrite

python run_grpo.py \
    --model Qwen/Qwen3-8B \
    --dataset ahans1/code_patrol_postprocessed \
    --data_download_dir data/code_patrol \
    --subset v7 --lr 1e-6 --batch_size 4 --rollout_batch_size 16 --vllm_model_shards 2 \
    --vllm_cache_utilization 0.6 \
    --num_generations 3 --max_prompt_length 8192 --max_response_length 1024 \
    --no-offload_weights_and_states \
    --wandb_entity azheng15-umd \
    --wandb_project DynaGuard2 \
    --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
    --resume_training \
    --overwrite

python run_grpo.py \
    --model Qwen/Qwen3-8B \
    --dataset ahans1/code_patrol_postprocessed \
    --data_download_dir data/code_patrol \
    --subset v8 --lr 1e-6 --batch_size 4 --rollout_batch_size 16 --vllm_model_shards 2 \
    --vllm_cache_utilization 0.6 \
    --num_generations 3 --max_prompt_length 8192 --max_response_length 1024 \
    --no-offload_weights_and_states \
    --wandb_entity azheng15-umd \
    --wandb_project DynaGuard2 \
    --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
    --resume_training \
    --overwrite

python run_grpo.py \
    --model Qwen/Qwen3-8B \
    --dataset ahans1/code_patrol_postprocessed \
    --data_download_dir data/code_patrol \
    --subset v9 --lr 1e-6 --batch_size 4 --rollout_batch_size 16 --vllm_model_shards 2 \
    --vllm_cache_utilization 0.6 \
    --num_generations 3 --max_prompt_length 8192 --max_response_length 1024 \
    --no-offload_weights_and_states \
    --wandb_entity azheng15-umd \
    --wandb_project DynaGuard2 \
    --checkpoint_dir /fs/cml-projects/guardian_models/verl/Andrew_runs \
    --resume_training \
    --overwrite

