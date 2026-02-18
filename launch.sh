# launch launch.sh --classical_logfile_names --gpu_type rtxa6000 --mem 100 --timelimit 48
# launch launch.sh --classical_logfile_names --gpu_type rtxa5000 --mem 128 --timelimit 24 --gpus 2

# --vllm_model_shards 2
# --batch_size_per_gpu 1 

python run_grpo.py \
    --model DynaGuard/DynaGuard-8B-6750 \
    --dataset ahans1/code_patrol_postprocessed \
    --data_download_dir data/code_patrol \
    --subset v2 --lr 1e-6 --batch_size 12 --rollout_batch_size 48 --vllm_model_shards 2 \
    --num_generations 12 --max_prompt_length 8192 --max_response_length 1024 \
    --no-offload_weights_and_states