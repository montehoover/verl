import multiprocessing
import argparse, os, subprocess
import torch
import wandb
from verl.tomlab.helpers import preprocess_dataset_gsm8k, get_short_model_name, get_last_checkpoint_path

def main(args):
    #########################################################
    # Setup
    #########################################################
    model_name = get_short_model_name(args.model)
    dataset_name = args.dataset.split("/")[-1] if "/" in args.dataset else args.dataset
    run_name = f"{model_name}_{dataset_name}_lr{args.lr}_bs{args.batch_size}"
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs found. Double check the that this is being called where you expect it to be."  
    if args.rollout_batch_size is None:
        rollout_batch_size = args.batch_size
    elif args.rollout_batch_size < args.batch_size:
        print(f"Warning: rollout_batch_size ({args.rollout_batch_size}) is less than batch_size ({args.batch_size}). Setting rollout_batch_size to {args.rollout_batch_size}")
        rollout_batch_size = args.rollout_batch_size
    else:
        rollout_batch_size = args.batch_size

    if args.use_wandb:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        logger_entries = ["console", "wandb"]
    else:
        logger_entries = ["console"]

    if args.resume_training:
        resume_mode = "auto"
    else:
        resume_mode = "disable"


    #########################################################
    # Dataset
    #########################################################
    train_files, val_files = preprocess_dataset_gsm8k(hf_dataset_name=args.dataset, local_save_dir=args.data_download_dir)


    #########################################################
    # Train
    #########################################################
    print("\nStarting GRPO...\n")
    grpo_cmd = [
        "python3",
        "-m", 
        "verl.trainer.main_ppo",
        # Admin Settings
        f"trainer.n_gpus_per_node={num_gpus}",
        f"trainer.nnodes={args.num_nodes}",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.test_freq={args.val_freq}",
        f"trainer.val_before_train={args.val_before_train}",
        f"trainer.max_actor_ckpt_to_keep={args.num_checkpoints_to_keep}",
        f"trainer.logger={logger_entries!r}",
        f"trainer.project_name={args.wandb_project}",
        f"trainer.experiment_name={run_name}",
        f"trainer.default_local_dir={args.checkpoint_dir}/{run_name}",
        f"trainer.resume_mode={resume_mode}",
        # Dataset
        f"data.train_files={train_files}",
        f"data.val_files={val_files}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"data.filter_overlong_prompts=True",
        f"data.truncation={args.truncation}",
        # Model
        f"actor_rollout_ref.model.path={args.model}",
        f"actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.model.lora_rank={0 if args.lora_rank is None else args.lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={0 if args.lora_rank is None else args.lora_alpha}",
        # Training Parameters
        f"actor_rollout_ref.actor.optim.lr={args.lr}",
        f"actor_rollout_ref.actor.optim.lr_scheduler_type={args.lr_schedule}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.batch_size}",
        f"data.train_batch_size={rollout_batch_size}",
        f"actor_rollout_ref.rollout.n={args.num_generations}",
        f"actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={args.kl_coef}",
        f"algorithm.adv_estimator=grpo",
        f"trainer.total_epochs={args.epochs}",
        # f"custom_reward_function.path=verl/compliance/helpers.py",
        # f"custom_reward_function.name=compute_reward",
        # Memory Management
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.model_shards}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.vllm_cache_utilization}",
        f"actor_rollout_ref.rollout.max_model_len={args.max_prompt_length + args.max_response_length}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={args.chunked_prefill}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={args.gradient_checkpointing}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={args.offload_ref_params}",
        f"actor_rollout_ref.actor.fsdp_config.offload_policy={args.offload_weights_and_states}",
        f"actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.actor.strategy=fsdp2", # Set to "fsdp" if using pytorch < 2.4
        f"actor_rollout_ref.ref.strategy=fsdp2",  # Set to "fsdp" if using pytorch < 2.4
    ]
    # Ensure that ROCR_VISIBLE_DEVICES is not set, otherwise it will conflict with CUDA_VISIBLE_DEVICES.
    # Ensure that the output is not buffered, so that we can see the output in real time.
    env = os.environ.copy()
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(grpo_cmd, check=True, env=env)

    last_checkpoint_path = get_last_checkpoint_path(run_name, checkpoint_dir=args.checkpoint_dir)
    print(f"\nSuccessfully completed GRPO. Last checkpoint was saved to {last_checkpoint_path}\n")
    wandb.finish()

    
def parse_args():
    parser = argparse.ArgumentParser(description="This script runs a PPO training process with configurable parameters.")
    
    # Admin Settings
    parser.add_argument("--use_wandb", default=True, action=argparse.BooleanOptionalAction, help="Enable Weights & Biases logging (default: enabled)")
    parser.add_argument("--wandb_entity", default="tomg-group-umd", help="Weights & Biases entity (default: tomg-group-umd)")
    parser.add_argument("--wandb_project", default="verl_demo", help="Weights & Biases project (default: verl_demo)")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory for checkpoints (default: checkpoints)")
    parser.add_argument("--exit_on_checkpoint_error", default=True, action=argparse.BooleanOptionalAction, help="Exit on checkpoint error (default: enabled)")
    parser.add_argument("--save_freq", default=20, type=int, help="At how many steps to save a checkpoint (default: 20)")
    parser.add_argument("--val_freq", default=5, type=int, help="At how many steps to run validation loop. Set to -1 to disable validation. (default: 5)")
    parser.add_argument("--val_before_train", default=False, action=argparse.BooleanOptionalAction, help="Run validation before training (default: disabled)")
    parser.add_argument("--num_checkpoints_to_keep", default=1, type=int, help="Number of checkpoints to keep. If None, I think they all are saved.")

    # Dataset
    parser.add_argument("--dataset", default="openai/gsm8k", help="Dataset name (default: openai/gsm8k)")
    parser.add_argument("--data_download_dir", default="data/verl_demo", help="Local directory for data (default: data/compliance)")
    parser.add_argument("--split", default=None, help="Dataset split (default: None)")
    parser.add_argument("--val_split", default=None, help="Validation dataset split (default: None)")
    parser.add_argument("--num_examples", type=int, default=-1, help="Number of examples to train on. -1 for all (default: -1)")
    parser.add_argument("--val_size", type=int, default=0.0, help="Fraction of examples for validation if val_split is not provided (default: 0.0)")
   
    # Model
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--lora_rank", default=None, type=int, help="LoRA rank. If None, LoRA is disabled (default: None)")
    parser.add_argument("--lora_alpha", default=None, type=int, help="LoRA alpha. If None, LoRA is disabled (default: None)")
    
    # Training Parameters
    # Note that Verl uses the number of dataset prompts (meaning rows) as its atomic unit for batch sizes. --rollout_batch_size controls the number of prompts used
    # to to generate a bunch of rollouts at once and the wandb update interval. This is regardless of the actual gradient update batch size (--batch_size) or 
    # the number of rollouts per prompt (--num_generations).
    # Further, the number of prompts in a gradient update is always based on --batch_size, but there will always be more sequences in the batch than prompts, because
    # each gradient update batch has batch_size * num_generations sequences.
    # The relationship between rollout_batch_size and batch_size is one that determines how much policy staleness there is. (Larger rollout_batch_size gives more
    # policy staleness, but faster training.)
    # In conclusion, for a fixed dataset size, changing batch_size will change the number of gradient updates, but it will not change the number of wandb updates.
    # And increasing num_generations will increase the number of sequences in a gradient update, but not the total number updates.
    # rollout_batch_size == num prompts in batch generation pass and wandb update (https://github.com/volcengine/verl/issues/1524)
    # batch_size == num prompts in a gradient update (https://github.com/volcengine/verl/issues/180)
    # batch_size * num_generations == num sequences in a gradient update (https://github.com/volcengine/verl/issues/180)
    parser.add_argument("--epochs", default=15, type=int, help="Number of epochs (default: 15)")
    parser.add_argument("--lr", default="1e-6", help="Learning rate (default: 1e-6)")
    parser.add_argument("--batch_size", default=256, type=int, help="Effective batch size for each gradient update (default: 256)")
    parser.add_argument("--rollout_batch_size", default=1024, type=int, help="Num prompts to use from dataset for batches of rollouts. If larger than batch_size, some of the rollouts will be off-policy. Larger values of rollout_batch_size trades speed for policy-closeness. (default: 1024)")
    parser.add_argument("--num_generations", default=5, type=int, help="Number of generations (default: 5)")
    parser.add_argument("--max_response_length", default=1024, type=int, help="Max response length (default: 1024)")
    parser.add_argument("--kl_coef", default=0.001, type=float, help="KL coefficient (default: 0.001)")
    parser.add_argument("--lr_schedule", default="cosine", choices=["constant", "cosine"], help="Learning rate schedule (default: cosine)", )
    # parser.add_argument("--algorithm", default="grpo", choices=["grpo", "drgrpo", "ppo"], help="Algorithm to use (default: grpo)")
    parser.add_argument("--resume_training", default=False, action=argparse.BooleanOptionalAction, help="Resume GRPO training from last checkpoint (default: False)")

    # Memory management
    parser.add_argument("--num_nodes", default=1, type=int, help="Number of nodes to we are using (default: 1)")
    parser.add_argument("--vllm_cache_utilization", default=0.6, type=float, help="VLLM cache utilization. Set very low if running out of GPU memory. (default: 0.6)")
    parser.add_argument("--model_shards", default=1, type=int, help="Number of model shards. Examples show using 2 when doing a 7B model on 8 GPUs. They use 4 when doing a 32B model on 32 GPUs. (default: 1)")
    parser.add_argument("--batch_size_per_gpu",default=1, type=int, help="Batch size per GPU. Reduce if hitting OOM. Increase for faster training. (default: 1)")
    parser.add_argument("--max_prompt_length", default=512, type=int, help="Remove samples from the dataset that are longer than this (default: 512)")
    parser.add_argument("--truncation", default="error", choices=["error", "left", "right", "middle"], help="Truncation behavior when prompts exceed max length. Should not get called if --max_prompt_length is set. (default: error)"),
    parser.add_argument("--chunked_prefill", default=True, action=argparse.BooleanOptionalAction, help="Enable chunked prefill in vllm. Trades memory savings for speed, and is True by default in Verl. (default: True)")
    parser.add_argument("--gradient_checkpointing", default=True, action=argparse.BooleanOptionalAction, help="Enable gradient checkpointing (recomputing activations during backward pass). Trades memory savings for speed, and is True by default in Verl. (default: True)")
    parser.add_argument("--offload_ref_params", default=True, action=argparse.BooleanOptionalAction, help="Offload the weights of the reference model (frozen version of model being trained). Trades memory savings for speed, and is True by default in Verl. (default: True)")
    parser.add_argument("--offload_weights_and_states", default=False, action=argparse.BooleanOptionalAction, help="FSDP2 native offload policy for model weights and optimizer states. Only works with FSDP2. If using FSDP, set to false.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)