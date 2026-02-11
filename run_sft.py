import argparse, os, shutil, subprocess
import torch
from verl.tomlab.helpers import get_short_model_name, get_lora_target_modules, LORA_TARGET_MODULE_CHOICES
from verl.tomlab.dataset_functions import preprocess_dataset

def main(args):
    #########################################################
    # Setup
    #########################################################
    model_name = get_short_model_name(args.model)
    dataset_name = args.dataset.split("/")[-1] if "/" in args.dataset else args.dataset
    run_name = f"{model_name}_{dataset_name}_sft_lr{args.lr}_bs{args.batch_size}"
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs found. Double check that this is being called where you expect it to be."

    lora_target_modules = get_lora_target_modules(args.lora_target_modules)

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
                print(f"Checkpoint directory already exists: {checkpoint_path}")
                print(f"  Use --overwrite to remove it and start fresh.")
                print(f"  Use --resume_training to continue from the last checkpoint.")
                raise SystemExit(1)

    if args.use_wandb:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        logger_entries = ["console", "wandb"]
    else:
        logger_entries = ["console"]

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

    # The SFT trainer divides train_batch_size by dp_size (num_gpus), then asserts
    # the result is divisible by micro_batch_size_per_gpu. So we need:
    #   batch_size % (num_gpus * micro_batch_size_per_gpu) == 0
    granularity = num_gpus * args.micro_batch_size_per_gpu

    if args.batch_size > num_train_examples:
        print(f"Note: batch_size ({args.batch_size}) > num training examples ({num_train_examples}). Reducing batch_size to {num_train_examples}.")
        args.batch_size = num_train_examples

    if args.batch_size % granularity != 0:
        args.batch_size = max(granularity, (args.batch_size // granularity) * granularity)
        print(f"Note: batch_size must be divisible by num_gpus * micro_batch_size_per_gpu ({granularity}). Adjusted to {args.batch_size}.")

    #########################################################
    # Build torchrun command
    #########################################################
    print(f"\nStarting SFT training...\n")
    sft_cmd = [
        "torchrun",
        "--standalone",
        f"--nnodes=1",
        f"--nproc_per_node={num_gpus}",
        "-m",
        "verl.trainer.fsdp_sft_trainer",
        # Admin Settings
        f"trainer.project_name={args.wandb_project}",
        f"trainer.experiment_name={run_name}",
        f"trainer.default_local_dir={checkpoint_path}",
        f"trainer.resume_mode={resume_mode}",
        f"trainer.logger={logger_entries!r}",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.test_freq={args.val_freq}",
        # Dataset
        f"data.train_files={train_files}",
        f"data.val_files={val_files or train_files}",
        f"data.prompt_key=extra_info",
        f"data.response_key=extra_info",
        "data.prompt_dict_keys=['question']",
        "+data.response_dict_keys=['answer']",
        # Model
        f"model.partial_pretrain={args.model}",
        f"model.lora_rank={0 if args.lora_rank is None else args.lora_rank}",
        f"model.lora_alpha={16 if args.lora_alpha is None else args.lora_alpha}",
        f"model.target_modules={lora_target_modules}",
        # Training parameters
        f"optim.lr={args.lr}",
        f"optim.lr_scheduler={args.lr_schedule}",
        f"trainer.total_epochs={args.epochs}",
        f"data.train_batch_size={args.batch_size}",
        # Memory management
        f"data.micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        f"model.enable_gradient_checkpointing={args.gradient_checkpointing}",
        f"model.strategy={args.strategy}",
        f"data.max_length={args.max_length}",
        f"data.truncation={args.truncation}",
    ]

    env = os.environ.copy()
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(sft_cmd, check=True, env=env)

    print(f"\nSFT training completed successfully.")
    print(f"Checkpoints saved to: {checkpoint_path}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="This script runs SFT (Supervised Fine-Tuning) with configurable parameters.")

    # Admin Settings
    parser.add_argument("--use_wandb", default=True, action=argparse.BooleanOptionalAction, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_entity", default="tomg-group-umd", help="Weights & Biases entity")
    parser.add_argument("--wandb_project", default="verl_demo_sft", help="Weights & Biases project")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--save_freq", default=-1, type=int, help="At how many steps to save a checkpoint. -1 to disable.")
    parser.add_argument("--val_freq", default=-1, type=int, help="At how many steps to run validation loop. -1 to disable.")
    parser.add_argument("--resume_training", default=False, action=argparse.BooleanOptionalAction, help="Resume from last checkpoint.")
    parser.add_argument("--overwrite", default=False, action=argparse.BooleanOptionalAction, help="Remove old checkpoint directory if it exists. Required to start fresh when checkpoints already exist.")

    # Dataset
    parser.add_argument("--dataset", default="openai/gsm8k", help="Dataset name")
    parser.add_argument("--subset", default=None, help="Dataset subset/config name")
    parser.add_argument("--val_split", default=None, help="Validation dataset split")
    parser.add_argument("--dataset_function", default="preprocess_gsm8k", help="Name of the dataset preprocessing function in dataset_functions.py")
    parser.add_argument("--data_download_dir", default="data/gsm8k", help="Local directory for data")
    parser.add_argument("--num_examples", type=int, default=-1, help="Number of examples to train on. -1 for all.")
    parser.add_argument("--val_size", type=float, default=0.0, help="Fraction of examples for validation if val_split is not provided")

    # Model
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--lora_rank", default=None, type=int, help="LoRA rank. If None, LoRA is disabled")
    parser.add_argument("--lora_alpha", default=None, type=int, help="LoRA alpha. If None, LoRA is disabled")
    parser.add_argument("--lora_target_modules", default="all-linear", choices=LORA_TARGET_MODULE_CHOICES, help="Target modules for LoRA matrices")

    # Training Parameters
    parser.add_argument("--epochs", default=4, type=int, help="Number of epochs")
    parser.add_argument("--lr", default="1e-5", help="Learning rate")
    parser.add_argument("--lr_schedule", default="cosine", choices=["cosine", "wsd"], help="Learning rate schedule")
    parser.add_argument("--batch_size", default=256, type=int, help="Global training batch size")

    # Memory management
    parser.add_argument("--gradient_checkpointing", default=True, action=argparse.BooleanOptionalAction, help="Enable gradient checkpointing")
    parser.add_argument("--strategy", default="fsdp2", choices=["fsdp", "fsdp2"], help="FSDP strategy. fsdp2 requires PyTorch >= 2.4")
    parser.add_argument("--max_length", default=1024, type=int, help="Max sequence length (prompt + response)")
    parser.add_argument("--truncation", default="error", choices=["error", "left", "right"], help="Truncation behavior when sequences exceed max length")
    parser.add_argument("--micro_batch_size_per_gpu", default=1, type=int, help="Micro batch size per GPU for gradient accumulation")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
