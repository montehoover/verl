import argparse, os, subprocess, sys
import torch
from requests import HTTPError
from verl.compliance.helpers import configure_logging, get_last_checkpoint_path, get_model_name, prepare_dataset_for_verl, push_to_hub, run_subprocess

def main(args):
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    model_path = args.model
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs found. Double check the that this is being called where you expect it to be."
    train_files, val_files = prepare_dataset_for_verl(
        dataset_path=args.dataset,
        subset=args.subset,
        split=args.split,
        num_examples=args.num_examples,
        num_val_examples=args.num_val_examples,
        redownload=args.redownload,
        local_dir="data/compliance",
    )

    ################################
    # SFT Training
    ################################
    if args.run_sft:
        print("Starting SFT...")
        model_name = get_model_name(model_path)
        sft_run_name = f"{model_name}_{args.split}_sft_lr{args.sft_lr}_bs{args.sft_batch_size}-epochs{args.sft_epochs}-examples{args.num_examples}"
        sft_cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            f"-m", "verl.trainer.fsdp_sft_trainer",
            f"model.partial_pretrain={model_path}",
            f"model.enable_gradient_checkpointing=True",
            f"data.train_files={train_files}",
            f"data.val_files={val_files}",
            f"data.prompt_key=extra_info",
            f"data.response_key=extra_info",
            f"data.prompt_dict_keys=['question']",
            f"data.response_dict_keys=['answer']",
            f"+data.add_system_prompt=True",
            f"data.max_length={args.max_prompt_length}",
            f"data.micro_batch_size_per_gpu={args.batch_size_per_gpu}",
            f"data.train_batch_size={args.sft_batch_size}",
            f"optim.lr={args.sft_lr}",
            f"trainer.total_epochs={args.sft_epochs}",
            f"trainer.logger=['console','wandb']",
            f"trainer.project_name={args.sft_wandb_project}",
            f"trainer.experiment_name={sft_run_name}",
            f"trainer.default_local_dir={args.checkpoint_dir}/{sft_run_name}",
        ]
        subprocess.run(sft_cmd, check=True)
        last_checkpoint_path = get_last_checkpoint_path(sft_run_name)
        print(f"Successfully completed SFT. Last checkpoint was saved to {last_checkpoint_path}")
        
        # Push to Hugging Face Hub
        try:
            hf_hub_path = push_to_hub(checkpoint_path=last_checkpoint_path, run_name=sft_run_name)
            print(f"Model pushed to Hugging Face Hub at {hf_hub_path}")
            # For use if continuing to GRPO
            model_path = hf_hub_path
        except HTTPError as e:
            print(f"There was an erro when pushing to hf hub: {e}")
            if args.exit_on_checkpoint_error:
                raise
            else:
                print("Continuing without pushing to Hugging Face Hub...")


    ################################
    # GRPO Training
    ################################
    if args.run_grpo:
        print("Starting GRPO...")
        model_name = get_model_name(model_path)
        grpo_run_name = f"{model_name}_{args.split}_grpo_lr{args.sft_lr}_bs{args.sft_batch_size}-epochs{args.grpo_epochs}"
        grpo_cmd = [
            "python3",
            "-m", "verl.trainer.main_ppo",
            f"algorithm.adv_estimator=grpo",
            f"data.train_files={train_files}",
            f"data.val_files={val_files}",
            f"data.train_batch_size={args.grpo_batch_size}",
            f"data.max_prompt_length={args.max_prompt_length}",
            f"data.max_response_length={args.max_response_length}",
            f"data.filter_overlong_prompts=True",
            f"data.truncation='error'",
            f"actor_rollout_ref.model.path={model_path}",
            f"actor_rollout_ref.actor.optim.lr={args.grpo_lr}",
            f"actor_rollout_ref.model.use_remove_padding=True",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={args.num_generations}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
            f"actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef=0.001",
            f"actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            f"actor_rollout_ref.actor.entropy_coeff=0",
            f"actor_rollout_ref.model.enable_gradient_checkpointing=True",
            f"actor_rollout_ref.actor.fsdp_config.param_offload=False",
            f"actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size=2", # Verl uses 2 when doing a 7B model on 8 GPUs. They use 4 when doing a 32B model on 32 GPUs. 
            f"actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
            f"actor_rollout_ref.rollout.n={args.num_generations}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
            f"actor_rollout_ref.ref.fsdp_config.param_offload=True",
            f"actor_rollout_ref.rollout.enable_chunked_prefill=False",
            f"algorithm.use_kl_in_reward=False",
            f"trainer.critic_warmup=0",
            f"trainer.logger=['console','wandb']",
            f"trainer.project_name={args.grpo_wandb_project}",
            f"trainer.experiment_name={grpo_run_name}",
            f"trainer.n_gpus_per_node={num_gpus}",
            f"trainer.nnodes=1",
            f"trainer.save_freq=20",
            f"trainer.test_freq=5",
            f"trainer.total_epochs={args.grpo_epochs}",
            f"trainer.default_local_dir=checkpoints/{grpo_run_name}",
            f"custom_reward_function.path=verl/compliance/helpers.py", 
            f"custom_reward_function.name=compute_reward", 
        ]
        subprocess.run(grpo_cmd, check=True)
        print("Successfully completed GRPO.")

    print("Training process completed.")

    # try:
    #     last_checkpoint_path = get_last_checkpoint_path(grpo_run_name)
    #     hf_hub_path = push_to_hub(checkpoint_path=last_checkpoint_path, run_name=grpo_run_name)
    #     print(f"Model pushed to Hugging Face Hub at {hf_hub_path}")
    #     # For use if continuing to GRPO
    #     model_path = hf_hub_path
    # except HTTPError as e:
    #     print(f"There was an erro when pushing to hf hub: {e}")
    #     if args.exit_on_checkpoint_error:
    #         raise
    #     else:
    #         print("Continuing without pushing to Hugging Face Hub...")

def parse_args():
    parser = argparse.ArgumentParser(description="This script runs a PPO training process with configurable parameters.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name (default: Qwen/Qwen3-0.6B)")
    
    # Dataset
    parser.add_argument("--dataset", default="tomg-group-umd/compliance", help="Dataset name (default: tomg-group-umd/compliance)")
    parser.add_argument("--subset", default="compliance", help="Dataset subset (default: compliance)")
    parser.add_argument("--split", default="train_32000_mix", help="Dataset split (default: train_cot)")
    parser.add_argument("--num_examples", type=int, default=-1, help="Number of examples to train on. -1 for all (default: 1000)")
    parser.add_argument("--num_val_examples", type=int, default=256, help="Number of examples for validation (default: 100)")
    parser.add_argument("--redownload", default=True, action=argparse.BooleanOptionalAction, help="Force redownload of data (default: disabled)")
    parser.add_argument("--max_prompt_length", default=8192, type=int, help="Max prompt length (default: 512)")
    parser.add_argument("--batch_size_per_gpu",default=2, type=int, help="Batch size per GPU (default: 2)")
    
    # Run info
    parser.add_argument("--run_sft", default=False, action=argparse.BooleanOptionalAction, help="Run SFT (default: enabled)")
    parser.add_argument("--run_grpo", default=True, action=argparse.BooleanOptionalAction, help="Run GRPO (default: enabled)")
    parser.add_argument("--download_local_dir", default="data/compliance", help="Local directory for data (default: data/compliance)")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory for checkpoints (default: checkpoints)")
    parser.add_argument("--exit_on_checkpoint_error", default=True, action=argparse.BooleanOptionalAction, help="Exit on checkpoint error (default: enabled)")

    # SFT
    parser.add_argument("--sft_wandb_project", default="sft-compliance", help="Trainer project name for WandB (default: sft-compliance)")
    parser.add_argument("--sft_epochs", default=1, type=int, help="Number of epochs (default: 4)")
    parser.add_argument("--sft_lr", default="1e-5", help="Learning rate (default: 1e-5)")
    parser.add_argument("--sft_batch_size", default=32, type=int, help="Total batch size (default: 32)")
    parser.add_argument("--wandb_entity", default="guardian-models", help="Weights & Biases entity (default: guardian-models)")

    # GRPO
    parser.add_argument("--grpo_wandb_project", default="grpo-compliance", help="Trainer project name for WandB (default: grpo-compliance)")
    parser.add_argument("--grpo_epochs", default=1, type=int, help="Number of epochs (default: 1)")
    parser.add_argument("--grpo_lr", default="1e-6", help="Learning rate (default: 1e-6)")
    parser.add_argument("--grpo_batch_size", default=48, type=int, help="Total batch size (default: 48)")
    parser.add_argument("--num_generations", default=12, type=int, help="Number of generations (default: 12)")
    parser.add_argument("--max_response_length", default=512, type=int, help="Max response length (default: 512)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)