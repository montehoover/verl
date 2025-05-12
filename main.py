import argparse, os, subprocess, sys
import torch
from requests import HTTPError
from verl.compliance.helpers import configure_logging, get_last_checkpoint_path, prepare_dataset_for_verl, push_to_hub, run_subprocess

def main(args):
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    model_name = args.model
    run_name = f"{model_name.split('/')[-1]}_{args.split}_lr{args.lr}_bs{args.batch_size}-epochs{args.epochs}"
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs found. Double check the that this is being called where you expect it to be."
    train_files, val_files = prepare_dataset_for_verl(
        dataset=args.dataset,
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
        sft_cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            f"-m", "verl.trainer.fsdp_sft_trainer",
            f"model.partial_pretrain={model_name}",
            f"model.enable_gradient_checkpointing=True",
            f"data.train_files={train_files}",
            f"data.val_files={val_files}",
            f"data.prompt_key=verl_stuff",
            f"data.response_key=verl_stuff",
            f"data.prompt_dict_keys=['question']",
            f"data.response_dict_keys=['answer']",
            f"+data.add_system_prompt=True",
            f"data.max_length=8192",
            f"data.micro_batch_size_per_gpu={args.batch_size_per_gpu}",
            f"data.train_batch_size={args.batch_size}",
            f"optim.lr={args.lr}",
            f"trainer.total_epochs={args.epochs}",
            f"trainer.logger=['console','wandb']",
            f"trainer.project_name={args.wandb_project_name}",
            f"trainer.experiment_name={run_name}",
            f"trainer.default_local_dir={args.checkpoint_dir}/{run_name}"
        ]
        subprocess.run(sft_cmd, check=True)
        last_checkpoint_path = get_last_checkpoint_path(run_name)
        print(f"Successfully completed SFT. Last checkpoint was saved to {last_checkpoint_path}")
        
        # Push to Hugging Face Hub
        try:
            hf_hub_path = push_to_hub(model_path=last_checkpoint_path, run_name=run_name)
            print(f"Model pushed to Hugging Face Hub at {hf_hub_path}")
            # For use if continuing to GRPO
            model_name = hf_hub_path
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
        grpo_cmd = [
            "python3",
            "-m", "verl.trainer.main_ppo",
            f"algorithm.adv_estimator=grpo",
            f"data.train_files={train_files}",
            f"data.val_files={val_files}",
            f"data.train_batch_size={args.batch_size}",
            f"data.max_prompt_length=512",
            f"data.max_response_length=1024",
            f"data.filter_overlong_prompts=True",
            f"data.truncation='error'",
            f"actor_rollout_ref.model.path={model_name}",
            f"actor_rollout_ref.actor.optim.lr=1e-6",
            f"actor_rollout_ref.model.use_remove_padding=True",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={args.batch_size}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
            f"actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef=0.001",
            f"actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            f"actor_rollout_ref.actor.entropy_coeff=0",
            f"actor_rollout_ref.model.enable_gradient_checkpointing=True",
            f"actor_rollout_ref.actor.fsdp_config.param_offload=False",
            f"actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={num_gpus}",
            f"actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
            f"actor_rollout_ref.rollout.n=5",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
            f"actor_rollout_ref.ref.fsdp_config.param_offload=True",
            f"algorithm.use_kl_in_reward=False",
            f"trainer.critic_warmup=0",
            f"trainer.logger=['console','wandb']",
            f"trainer.project_name={args.wandb_project_name}",
            f"trainer.experiment_name={run_name}",
            f"trainer.n_gpus_per_node={num_gpus}",
            f"trainer.nnodes=1",
            f"trainer.save_freq=20",
            f"trainer.test_freq=5",
            f"trainer.total_epochs={args.epochs}",
            f"trainer.default_local_dir=checkpoints/{run_name}"
        ]
        subprocess.run(grpo_cmd, check=True)
        print("Successfully completed GRPO.")

    print("Training process completed.")


def parse_args():
    parser = argparse.ArgumentParser(description="This script runs a PPO training process with configurable parameters.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--dataset", default="tomg-group-umd/compliance", help="Dataset name (default: tomg-group-umd/compliance)")
    parser.add_argument("--subset", default="compliance", help="Dataset subset (default: compliance)")
    parser.add_argument("--split", default="train_32000_mix", help="Dataset split (default: train_cot)")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs (default: 1)")
    parser.add_argument("--lr", default="1e-5", help="Learning rate (default: 1e-5)")
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size (default: 32)")
    parser.add_argument("--batch_size_per_gpu",default=2, type=int, help="Batch size per GPU (default: 2)")
    parser.add_argument("--wandb_entity", default="guardian-models", help="Weights & Biases entity (default: guardian-models)")
    parser.add_argument("--num_examples", type=int, default=-1, help="Number of examples to train on. -1 for all (default: 1000)")
    parser.add_argument("--download_local_dir", default="data/compliance", help="Local directory for data (default: data/compliance)")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory for checkpoints (default: checkpoints)")
    parser.add_argument("--num_val_examples", type=int, default=256, help="Number of examples for validation (default: 100)")
    parser.add_argument("--redownload", default=True, action=argparse.BooleanOptionalAction, help="Force redownload of data (default: disabled)")
    parser.add_argument("--wandb_project_name", default="sft-compliance", help="Trainer project name for WandB (default: sft-compliance)")
    parser.add_argument("--exit_on_checkpoint_error", default=True, action=argparse.BooleanOptionalAction, help="Exit on checkpoint error (default: enabled)")
    parser.add_argument("--run_sft", default=True, action=argparse.BooleanOptionalAction, help="Run SFT (default: enabled)")
    parser.add_argument("--run_grpo", default=False, action=argparse.BooleanOptionalAction, help="Run GRPO (default: enabled)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)