import argparse, os, shutil, subprocess, warnings
# Suppress future warnings and user warnings everywhere. Here is what gets suppressed:
# - Torch pynvml package is deprecated
# - Ray future warning about default behavior when num gpus environment variable is set to 0
# - Verl dataloader warning stating that default value of 8 workers might not be optimal
# - Verl tokenizer warning that always gets printed when not using a multimodal model
# - Verl warning stating that it is not using critic for GRPO
warning_filters = "ignore::UserWarning,ignore::FutureWarning"
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = warning_filters # For subprocesses

import torch
from verl.tomlab.helpers import get_short_model_name, get_last_checkpoint_path, get_lora_target_modules, LORA_TARGET_MODULE_CHOICES
from verl.tomlab.dataset_functions import preprocess_dataset

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
        print(f"Note: rollout_batch_size ({args.rollout_batch_size}) is less than batch_size ({args.batch_size}). Setting rollout_batch_size to batch_size ({args.batch_size})")
        rollout_batch_size = args.batch_size
    else:
        rollout_batch_size = args.rollout_batch_size

    lora_target_modules = get_lora_target_modules(args.lora_target_modules)

    if args.use_wandb:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        logger_entries = ["console", "wandb"]
    else:
        logger_entries = ["console"]

    checkpoint_path = f"{args.checkpoint_dir}/{run_name}"
    if args.resume_training:
        resume_mode = "auto"
    else:
        resume_mode = "disable"
        if os.path.exists(checkpoint_path):
            if args.overwrite:
                print(f"Note: Removing old checkpoint directory: {checkpoint_path}")
                shutil.rmtree(checkpoint_path)
            else:
                print("Checkpoint directory already exists: {checkpoint_path}")
                print(f"  Use --overwrite to remove it and start fresh.")
                print(f"  Use --resume_training to continue from the last checkpoint.")
                raise SystemExit(1)

    # See: https://verl.readthedocs.io/en/latest/algo/grpo.html#drgrpo
    if args.algorithm == "grpo":
        adv_estimator = "grpo"
        use_kl_loss = True
        norm_adv_by_std_in_grpo = True
        loss_agg_mode = "token-mean"
        ppo_stuff = []
    elif args.algorithm == "drgrpo":
        adv_estimator = "grpo"
        use_kl_loss = False
        norm_adv_by_std_in_grpo = False
        loss_agg_mode = "seq-mean-token-sum-norm"
        ppo_stuff = []
    elif args.algorithm == "ppo":
        adv_estimator = "gae"
        use_kl_loss = False
        norm_adv_by_std_in_grpo = "null"
        loss_agg_mode = "token-mean"
        ppo_stuff = [
            f"algorithm.kl_ctrl.kl_coef={args.kl_coef}",
            f"critic.model.path={args.model}",
            f"critic.optim.lr={args.lr * 10}",
            f"critic.ppo_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
            f"critic.model.enable_gradient_checkpointing={args.gradient_checkpointing}",
        ]

    #########################################################
    # Dataset
    #########################################################
    train_files, val_files, num_train_examples = preprocess_dataset(
        map_fn_name=args.dataset_function,
        hf_dataset_name=args.dataset,
        hf_dataset_subset=args.subset,
        local_save_dir=args.data_download_dir,
        num_examples=args.num_examples,
        val_split=args.val_split,
        val_size=args.val_size,
    )
    if val_files is None:
        args.val_freq = -1
    if args.batch_size > num_train_examples:
        print(f"Note: batch_size ({args.batch_size}) > num training examples ({num_train_examples}). Reducing batch_size to {num_train_examples}.")
        args.batch_size = num_train_examples
    if rollout_batch_size > num_train_examples:
        print(f"Note: rollout_batch_size ({rollout_batch_size}) > num training examples ({num_train_examples}). Reducing rollout_batch_size to {num_train_examples}.")
        rollout_batch_size = num_train_examples

    # verl's agent loop chunks rollout sequences evenly across workers.
    # Ensure rollout_batch_size is divisible by num_workers.
    if rollout_batch_size % args.num_workers != 0:
        rollout_batch_size = (rollout_batch_size // args.num_workers) * args.num_workers or args.num_workers
        print(f"Note: rollout_batch_size must be divisible by num_workers ({args.num_workers}). Reducing to {rollout_batch_size}.")
    if args.batch_size > rollout_batch_size:
        args.batch_size = rollout_batch_size
        print(f"Note: batch_size reduced to {args.batch_size} to match rollout_batch_size.")


    #########################################################
    # Train
    #########################################################
    print(f"\nStarting {args.algorithm.upper()}...\n")
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
        f"trainer.default_local_dir={checkpoint_path}",
        f"trainer.resume_mode={resume_mode}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONWARNINGS='{warning_filters}'",
        # Dataset
        f"data.train_files={train_files}",
        f"data.val_files={val_files or train_files}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"data.filter_overlong_prompts=True",
        f"data.truncation={args.truncation}",
        # Model
        f"actor_rollout_ref.model.path={args.model}",
        f"actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.model.lora_rank={0 if args.lora_rank is None else args.lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={0 if args.lora_rank is None else args.lora_alpha}",
        f"actor_rollout_ref.model.target_modules={lora_target_modules}",
        # Training Parameters
        f"actor_rollout_ref.actor.optim.lr={args.lr}",
        f"actor_rollout_ref.actor.optim.lr_scheduler_type={args.lr_schedule}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.batch_size}",
        f"data.train_batch_size={rollout_batch_size}",
        f"actor_rollout_ref.rollout.n={args.num_generations}",
        f"algorithm.adv_estimator={adv_estimator}",
        f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
        f"actor_rollout_ref.actor.kl_loss_coef={args.kl_coef}",
        f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}",
        f"algorithm.norm_adv_by_std_in_grpo={norm_adv_by_std_in_grpo}",
        f"trainer.total_epochs={args.epochs}",
        f"custom_reward_function.path=verl/tomlab/reward_functions.py",
        f"custom_reward_function.name={args.reward_function}",
        # Memory Management
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.vllm_model_shards}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.vllm_cache_utilization}",
        f"actor_rollout_ref.rollout.max_model_len={args.max_prompt_length + args.max_response_length}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={args.chunked_prefill}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={args.gradient_checkpointing}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={args.offload_ref_params}",
        f"actor_rollout_ref.actor.fsdp_config.offload_policy={args.offload_weights_and_states}",
        f"actor_rollout_ref.rollout.agent.num_workers={args.num_workers}",
        f"actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes={args.update_weights_bucket_mb}",
        f"actor_rollout_ref.actor.strategy=fsdp2", # Set to "fsdp" if using pytorch < 2.4
        f"actor_rollout_ref.ref.strategy=fsdp2",  # Set to "fsdp" if using pytorch < 2.4
    ] + ppo_stuff
    # Ensure that ROCR_VISIBLE_DEVICES is not set, otherwise it will conflict with CUDA_VISIBLE_DEVICES.
    # Ensure that the output is not buffered, so that we can see the output in real time.
    env = os.environ.copy()
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(grpo_cmd, check=True, env=env)

    last_checkpoint_path = get_last_checkpoint_path(run_name, checkpoint_dir=args.checkpoint_dir)
    print("Run completed sucessfully if we have gotten to this point. If you see an error from teardown_atexit, that is expected because Ray is trying to shutdown workers while wandb is still running.")
    print(f"\nSuccessfully completed {args.algorithm.upper()}. Last checkpoint was saved to {last_checkpoint_path}\n")

    
def parse_args():
    parser = argparse.ArgumentParser(description="This script runs a PPO training process with configurable parameters.")
    
    # Admin Settings
    parser.add_argument("--use_wandb", default=True, action=argparse.BooleanOptionalAction, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_entity", default="tomg-group-umd", help="Weights & Biases entity")
    parser.add_argument("--wandb_project", default="verl_demo", help="Weights & Biases project")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--exit_on_checkpoint_error", default=True, action=argparse.BooleanOptionalAction, help="Exit on checkpoint error")
    parser.add_argument("--save_freq", default=20, type=int, help="At how many steps to save a checkpoint")
    parser.add_argument("--val_freq", default=5, type=int, help="At how many steps to run validation loop. Set to -1 to disable validation.")
    parser.add_argument("--val_before_train", default=False, action=argparse.BooleanOptionalAction, help="Run validation before training")
    parser.add_argument("--num_checkpoints_to_keep", default=1, type=int, help="Number of checkpoints to keep. If None, I think they all are saved.")
    parser.add_argument("--overwrite", default=False, action=argparse.BooleanOptionalAction, help="Remove old checkpoint directory if it exists. Required to start fresh when checkpoints already exist.")

    # Dataset
    parser.add_argument("--dataset", default="openai/gsm8k", help="Dataset name")
    parser.add_argument("--subset", default=None, help="Dataset subset/config name")
    parser.add_argument("--split", default=None, help="Dataset split")
    parser.add_argument("--val_split", default=None, help="Validation dataset split")
    parser.add_argument("--data_download_dir", default="data/gsm8k", help="Local directory for data")
    parser.add_argument("--dataset_function", default="preprocess_gsm8k", help="Name of the dataset preprocessing function in dataset_functions.py")
    parser.add_argument("--reward_function", default="gsm8k_reward", help="Reward function to use")
    parser.add_argument("--num_examples", type=int, default=-1, help="Number of examples to train on. -1 for all.")
    parser.add_argument("--val_size", type=float, default=0.0, help="Fraction of examples for validation if val_split is not provided")
    
    # Model
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--lora_rank", default=None, type=int, help="LoRA rank. If None, LoRA is disabled")
    parser.add_argument("--lora_alpha", default=None, type=int, help="LoRA alpha. If None, LoRA is disabled")
    parser.add_argument("--lora_target_modules", default="all-linear", choices=LORA_TARGET_MODULE_CHOICES, help="Target modules for LoRA matrices")
    # LORA_TARGET_MODULE_CHOICES: ["all-linear", "all-linear-and-embedding", "all-attention", "qv-only"]


    # Training Parameters
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument("--lr", default="1e-6", help="Learning rate")
    parser.add_argument("--batch_size", default=256, type=int, help="Effective batch size for each gradient update")
    parser.add_argument("--rollout_batch_size", default=1024, type=int, help="Num prompts to use from dataset for batches of rollouts. If larger than batch_size, some of the rollouts will be off-policy. Larger values of rollout_batch_size trades speed for policy-closeness.")
    parser.add_argument("--num_generations", default=5, type=int, help="Number of generations")
    parser.add_argument("--max_response_length", default=1024, type=int, help="Max response length")
    parser.add_argument("--kl_coef", default=0.001, type=float, help="KL coefficient")
    parser.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine"], help="Learning rate schedule")
    parser.add_argument("--algorithm", default="grpo", choices=["grpo", "drgrpo", "ppo"], help="Algorithm to use")
    parser.add_argument("--resume_training", default=False, action=argparse.BooleanOptionalAction, help="Resume GRPO training from last checkpoint")

    # Memory management
    parser.add_argument("--num_nodes", default=1, type=int, help="Number of nodes to we are using")
    parser.add_argument("--vllm_cache_utilization", default=0.6, type=float, help="VLLM cache utilization. Set very low if running out of GPU memory.")
    parser.add_argument("--vllm_model_shards", default=1, type=int, help="Number of model shards for Vllm. Set as low as possible given the model size for fastest generations. Examples show using 2 when doing a 7B model on 8 GPUs. They use 4 when doing a 32B model on 32 GPUs.")
    parser.add_argument("--batch_size_per_gpu",default=1, type=int, help="Batch size per GPU. Reduce if hitting OOM. Increase for faster training.")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of CPU agent loop workers that orchestrate rollout generation. Verl defaults to 8. rollout_batch_size must be divisible by this.")
    parser.add_argument("--max_prompt_length", default=512, type=int, help="Remove samples from the dataset that are longer than this")
    parser.add_argument("--truncation", default="error", choices=["error", "left", "right", "middle"], help="Truncation behavior when prompts exceed max length. Should not get called if --max_prompt_length is set."),
    parser.add_argument("--chunked_prefill", default=True, action=argparse.BooleanOptionalAction, help="Enable chunked prefill in vllm. Trades memory savings for speed, and is True by default in Verl.")
    parser.add_argument("--gradient_checkpointing", default=True, action=argparse.BooleanOptionalAction, help="Enable gradient checkpointing (recomputing activations during backward pass). Trades memory savings for speed, and is True by default in Verl.")
    parser.add_argument("--offload_ref_params", default=True, action=argparse.BooleanOptionalAction, help="Offload the weights of the reference model (frozen version of model being trained). Trades memory savings for speed, and is True by default in Verl.")
    parser.add_argument("--offload_weights_and_states", default=False, action=argparse.BooleanOptionalAction, help="FSDP2 native offload policy for model weights and optimizer states. Only works with FSDP2. If using FSDP, set to false.")
    parser.add_argument("--update_weights_bucket_mb", default=4096, type=int, help="Size in MB of the bucket for transferring weights from FSDP2 to vLLM. Must be larger than the biggest single parameter in fp32. Default 4096 accommodates 8B models. Verl default is 2048, and that worked with FSDP1, but not FSDP2.")

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)