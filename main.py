import argparse, os, subprocess, sys
import torch
from verl.compliance.helpers import configure_logging, prepare_dataset_for_verl, run_subprocess

def main(args):
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    run_name = f"{args.model.split('/')[-1]}-{args.split}-lr{args.lr}-bs{args.batch_size}-epochs{args.epochs}"
    num_gpus = torch.cuda.device_count()
    train_files, val_files = prepare_dataset_for_verl(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        num_examples=args.num_examples,
        redownload=args.redownload,
        local_dir="data/compliance",
    )

    ################################
    # SFT Training
    ################################
    print("Starting training...")
    torchrun_cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        f"-m", "verl.trainer.fsdp_sft_trainer",
        f"model.partial_pretrain={args.model}",
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
        f"trainer.default_local_dir=checkpoints/{run_name}"
    ]
    subprocess.run(torchrun_cmd, check=True)
    # run_subprocess(torchrun_cmd, logger, check=True)
    print(f"Training completed successfully. Checkpoints saved to checkpoints/{run_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="This script runs a PPO training process with configurable parameters.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--dataset", default="tomg-group-umd/compliance", help="Dataset name (default: tomg-group-umd/compliance)")
    # parser.add_argument("--dataset", default="montehoover/compliance", help="Dataset name (default: tomg-group-umd/compliance)")
    parser.add_argument("--subset", default="compliance", help="Dataset subset (default: compliance)")
    parser.add_argument("--split", default="train_32000_mix", help="Dataset split (default: train_cot)")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs (default: 1)")
    parser.add_argument("--lr", default="1e-5", help="Learning rate (default: 1e-5)")
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size (default: 32)")
    parser.add_argument("--batch_size_per_gpu",default=2, type=int, help="Batch size per GPU (default: 2)")
    parser.add_argument("--wandb_entity", default="guardian-models", help="Weights & Biases entity (default: guardian-models)")
    parser.add_argument("--num_examples", type=int, default=-1, help="Number of examples to train on. -1 for all (default: 1000)")
    parser.add_argument("--download_local_dir", default="data/compliance", help="Local directory for data (default: data/compliance)")
    parser.add_argument("--redownload", default=True, action=argparse.BooleanOptionalAction, help="Force redownload of data (default: disabled)")
    parser.add_argument("--wandb_project_name", default="sft-compliance", help="Trainer project name for WandB (default: sft-compliance)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)