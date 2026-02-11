# Run with:
# launch launch.sh --classical_logfile_names --gpu_type rtxa6000 --gpus 4 --mem 120

# This reproduces the results from https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-8b.sh
python run_grpo.py --model Qwen/Qwen3-8B --dataset openai/gsm8k --val_split test --epochs 15 --lr 1e-6 --batch_size 256 --rollout_batch_size 1024 --num_generations 5 --max_prompt_length 512 --max_response_length 1024 --kl_coef 0.001 --lr_schedule constant --save_freq 20 --val_freq 5 --vllm_cache_utilization 0.4 --batch_size_per_gpu 1 --vllm_model_shards 2 --no-offload_weights_and_states --update_weights_bucket_mb 4096
