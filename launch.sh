# Run with:
# launch launch.sh --classical_logfile_names --gpu_type rtxa6000 --gpus 2 --mem 30

# This reproduces the results from https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-8b.sh
python run_grpo.py --model Qwen/Qwen3-8B --dataset openai/gsm8k --epochs 15 --lr 1e-6 --batch_size 256 --rollout_batch_size 1024 --num_generations 5 --max_response_length 1024 --kl_coef 0.001 --lr_schedule constant --save_freq 20 --val_freq 5 --vllm_cache_utilization 0.6 --batch_size_per_gpu 1 --max_prompt_length 512
    