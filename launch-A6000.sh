# launch launch.sh --classical_logfile_names --gpu_type rtxa6000 --mem 123 --timelimit 48 --gpus 2

python run_grpo.py \
    --model DynaGuard/DynaGuard-8B-6750 \
    --dataset ahans1/code_patrol_postprocessed \
    --data_download_dir data/code_patrol \
    --subset v2 --lr 1e-6 --batch_size 4 --rollout_batch_size 16 --vllm_model_shards 2 \
    --vllm_cache_utilization 0.5 \
    --num_generations 3 --max_prompt_length 8192 --max_response_length 1024 \
    --no-offload_weights_and_states \
    --wandb_entity azheng15-umd \
    --wandb_project verl

