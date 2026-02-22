#!/usr/bin/env python3
"""
DynaGuard SFT Training on Modal

Simple SFT training script with automatic GPU detection and LoRA support.

Usage:
    # Quick test with 10 examples
    modal run deployment/dynaguard_modal_sft.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset openai/gsm8k --num-examples 10

    # Full training with LoRA (default)
    modal run deployment/dynaguard_modal_sft.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset openai/gsm8k \
        --lora-rank 8 --lora-alpha 16 --lora-target-modules all-linear

    # Full-parameter SFT (no LoRA)
    modal run deployment/dynaguard_modal_sft.py --model Qwen/Qwen3-8B --dataset openai/gsm8k --no-lora

    # With CPU offloading for limited GPU memory
    modal run deployment/dynaguard_modal_sft.py --model Qwen/Qwen3-8B --dataset openai/gsm8k --offload

LoRA target modules options:
    - all-linear (default): Apply LoRA to all linear layers
    - all-linear-and-embedding: Also applies LoRA to embedding layer
    - all-attention: Only attention layers
    - qv-only: Only query and value projections
"""

import subprocess
from pathlib import Path
from typing import Optional

import modal

# ## App and Image Setup

app = modal.App("dynaguard-sft")

VERL_REPO_PATH: Path = Path("/root/verl")

image = (
    modal.Image.from_registry("verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1")
    .apt_install("git")
    .run_commands(f"git clone https://github.com/volcengine/verl {VERL_REPO_PATH}")
)

# ## Volumes

DATA_PATH: Path = Path("/data")
MODELS_PATH: Path = Path("/models")
MINUTES: int = 60

data_volume = modal.Volume.from_name("dynaguard-sft-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("dynaguard-sft-checkpoints", create_if_missing=True)


# ## Dataset Preparation


@app.function(image=image, volumes={DATA_PATH: data_volume}, timeout=30 * MINUTES)
def prep_dataset(dataset: str = "openai/gsm8k", num_examples: int = -1) -> None:
    """
    Prepare dataset for training from HuggingFace.

    Downloads to /tmp first (Modal best practice), then saves to Volume.

    Args:
        dataset: HuggingFace dataset name (e.g., "openai/gsm8k")
        num_examples: Number of examples to use (-1 for all)
    """
    import shutil
    from pathlib import Path as LocalPath

    import pandas as pd
    from datasets import load_dataset

    # Download to /tmp first (Modal best practice for dataset ingestion)
    tmp_dir = LocalPath("/tmp/dataset_cache")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset}")

    if dataset == "openai/gsm8k":
        ds = load_dataset("openai/gsm8k", "main", cache_dir=str(tmp_dir))

        def format_example(example, idx, split):
            return {
                "data_source": "openai/gsm8k",
                "extra_info": {
                    "question": example["question"],
                    "answer": example["answer"],
                    "index": idx,
                    "split": split,
                },
            }

        train_data = [format_example(ex, i, "train") for i, ex in enumerate(ds["train"])]
        test_data = [format_example(ex, i, "test") for i, ex in enumerate(ds["test"])]

    else:
        # Generic HuggingFace dataset loading
        ds = load_dataset(dataset, cache_dir=str(tmp_dir))

        # Auto-detect column names
        sample = ds["train"][0]
        question_key = next((k for k in ["question", "prompt", "input", "text"] if k in sample), None)
        answer_key = next((k for k in ["answer", "response", "output", "target"] if k in sample), None)

        if not question_key or not answer_key:
            raise ValueError(f"Could not find question/answer columns. Found: {list(sample.keys())}")

        print(f"Using columns: question='{question_key}', answer='{answer_key}'")

        def format_generic(example, idx, split):
            return {
                "data_source": dataset,
                "extra_info": {
                    "question": example[question_key],
                    "answer": example[answer_key],
                    "index": idx,
                    "split": split,
                },
            }

        train_data = [format_generic(ex, i, "train") for i, ex in enumerate(ds["train"])]
        test_split = "test" if "test" in ds else "validation" if "validation" in ds else None
        test_data = [format_generic(ex, i, "test") for i, ex in enumerate(ds[test_split])] if test_split else []

    # Apply num_examples limit
    if num_examples > 0:
        train_data = train_data[:num_examples]
        test_data = test_data[: max(1, num_examples // 10)]

    # Save to /tmp first, then copy to Volume
    tmp_train = tmp_dir / "train.parquet"
    tmp_test = tmp_dir / "test.parquet"

    pd.DataFrame(train_data).to_parquet(tmp_train)
    pd.DataFrame(test_data).to_parquet(tmp_test)

    # Copy to Volume
    shutil.copy(tmp_train, DATA_PATH / "train.parquet")
    shutil.copy(tmp_test, DATA_PATH / "test.parquet")

    data_volume.commit()

    print(f"Saved {len(train_data)} train examples to {DATA_PATH / 'train.parquet'}")
    print(f"Saved {len(test_data)} test examples to {DATA_PATH / 'test.parquet'}")


# ## Training Function


def get_gpu_count(gpu_config: str) -> int:
    """Extract GPU count from Modal GPU config string (e.g., 'A100:2' -> 2)."""
    if ":" in gpu_config:
        return int(gpu_config.split(":")[1])
    return 1


def build_train_command(
    num_gpus: int,
    model: str,
    epochs: int,
    lr: float,
    micro_batch_size: int,
    max_length: int,
    # LoRA config
    use_lora: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_target_modules: str,
    # Memory config
    offload: bool,
    gradient_checkpointing: bool,
    # Logging
    project_name: str,
    experiment_name: str,
    save_freq: int,
) -> list[str]:
    """Build the torchrun command for SFT training."""
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={num_gpus}",
        "-m",
        "verl.trainer.fsdp_sft_trainer",
        f"data.train_files={DATA_PATH / 'train.parquet'}",
        f"data.val_files={DATA_PATH / 'test.parquet'}",
        "data.prompt_key=extra_info",
        "data.response_key=extra_info",
        "data.prompt_dict_keys=['question']",
        "+data.response_dict_keys=['answer']",
        f"data.micro_batch_size_per_gpu={micro_batch_size}",
        f"data.max_length={max_length}",
        f"model.partial_pretrain={model}",
        f"optim.lr={lr}",
        f"trainer.default_local_dir={MODELS_PATH}",
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={experiment_name}",
        "trainer.logger=['console']",
        f"trainer.total_epochs={epochs}",
        f"trainer.save_freq={save_freq}",
    ]

    # LoRA configuration
    if use_lora:
        cmd.extend(
            [
                f"model.lora_rank={lora_rank}",
                f"model.lora_alpha={lora_alpha}",
                f"model.target_modules={lora_target_modules}",
            ]
        )

    # Memory optimization
    if offload:
        cmd.extend(
            [
                "model.fsdp_config.cpu_offload=True",
                "model.fsdp_config.offload_params=True",
            ]
        )

    if gradient_checkpointing:
        cmd.append("model.enable_gradient_checkpointing=True")

    return cmd


@app.function(
    image=image,
    gpu="A100:2",
    volumes={
        MODELS_PATH: checkpoints_volume,
        DATA_PATH: data_volume,
    },
    timeout=24 * 60 * MINUTES,
)
def train(
    # Model config
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset: str = "openai/gsm8k",
    num_examples: int = -1,
    # Training config
    epochs: int = 3,
    lr: float = 1e-4,
    micro_batch_size: int = 4,
    max_length: int = 1024,
    # LoRA config
    no_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: str = "all-linear",
    # Memory config
    offload: bool = False,
    gradient_checkpointing: bool = True,
    # Logging config
    project_name: str = "dynaguard-sft",
    experiment_name: str = "sft-run",
    save_freq: int = -1,
) -> None:
    """
    Run SFT training with automatic GPU detection and LoRA support.

    Args:
        model: HuggingFace model name (e.g., Qwen/Qwen2.5-0.5B-Instruct)
        dataset: Dataset to use (default: openai/gsm8k)
        num_examples: Number of training examples (-1 for all)
        epochs: Number of training epochs
        lr: Learning rate
        micro_batch_size: Micro batch size per GPU
        max_length: Maximum sequence length
        no_lora: Disable LoRA (full-parameter fine-tuning)
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_target_modules: LoRA target modules (all-linear, all-linear-and-embedding, all-attention, qv-only)
        offload: Enable CPU offloading for weights and optimizer states
        gradient_checkpointing: Enable gradient checkpointing
        project_name: Project name for logging
        experiment_name: Experiment name for logging
        save_freq: Save checkpoint frequency (-1 for end only)
    """
    data_volume.reload()

    # Auto-detect GPU count from the function's GPU config
    num_gpus = 2  # Matches gpu="A100:2" above

    print(f"=== DynaGuard SFT Training ===")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"GPUs: {num_gpus}")
    print(f"LoRA: {'Disabled' if no_lora else f'rank={lora_rank}, alpha={lora_alpha}, targets={lora_target_modules}'}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"CPU Offload: {offload}")
    print()

    cmd = build_train_command(
        num_gpus=num_gpus,
        model=model,
        epochs=epochs,
        lr=lr,
        micro_batch_size=micro_batch_size,
        max_length=max_length,
        use_lora=not no_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_modules=lora_target_modules,
        offload=offload,
        gradient_checkpointing=gradient_checkpointing,
        project_name=project_name,
        experiment_name=experiment_name,
        save_freq=save_freq,
    )

    print("Running command:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)
    checkpoints_volume.commit()
    print("Training complete! Checkpoints saved.")


# ## High-memory variant for larger models


@app.function(
    image=image,
    gpu="A100-80GB:4",
    volumes={
        MODELS_PATH: checkpoints_volume,
        DATA_PATH: data_volume,
    },
    timeout=24 * 60 * MINUTES,
)
def train_large(
    model: str = "Qwen/Qwen3-8B",
    dataset: str = "openai/gsm8k",
    num_examples: int = -1,
    epochs: int = 3,
    lr: float = 1e-5,
    micro_batch_size: int = 2,
    max_length: int = 1024,
    no_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: str = "all-linear",
    offload: bool = False,
    gradient_checkpointing: bool = True,
    project_name: str = "dynaguard-sft",
    experiment_name: str = "sft-large-run",
    save_freq: int = -1,
) -> None:
    """
    Run SFT training for larger models (8B+) with 4x A100-80GB GPUs.
    Same parameters as train() but with more GPU memory.
    """
    data_volume.reload()

    num_gpus = 4  # Matches gpu="A100-80GB:4"

    print(f"=== DynaGuard SFT Training (Large Model) ===")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"GPUs: {num_gpus} x A100-80GB")
    print(f"LoRA: {'Disabled' if no_lora else f'rank={lora_rank}, alpha={lora_alpha}, targets={lora_target_modules}'}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"CPU Offload: {offload}")
    print()

    cmd = build_train_command(
        num_gpus=num_gpus,
        model=model,
        epochs=epochs,
        lr=lr,
        micro_batch_size=micro_batch_size,
        max_length=max_length,
        use_lora=not no_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_modules=lora_target_modules,
        offload=offload,
        gradient_checkpointing=gradient_checkpointing,
        project_name=project_name,
        experiment_name=experiment_name,
        save_freq=save_freq,
    )

    print("Running command:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)
    checkpoints_volume.commit()
    print("Training complete! Checkpoints saved.")


# ## CLI Entry Point


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset: str = "openai/gsm8k",
    num_examples: int = -1,
    epochs: int = 3,
    lr: float = 1e-4,
    micro_batch_size: int = 4,
    max_length: int = 1024,
    no_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: str = "all-linear",
    offload: bool = False,
    large: bool = False,
    project_name: str = "dynaguard-sft",
    experiment_name: str = "sft-run",
    save_freq: int = -1,
    prep_data: bool = False,
):
    """
    DynaGuard SFT Training CLI

    Examples:
        # Quick test
        modal run deployment/dynaguard_modal_sft.py --model Qwen/Qwen2.5-0.5B-Instruct --num-examples 10

        # With LoRA
        modal run deployment/dynaguard_modal_sft.py --lora-rank 8 --lora-alpha 16

        # Full-parameter (no LoRA)
        modal run deployment/dynaguard_modal_sft.py --no-lora

        # Large model with 4x A100-80GB
        modal run deployment/dynaguard_modal_sft.py --large --model Qwen/Qwen3-8B

        # Prepare dataset only
        modal run deployment/dynaguard_modal_sft.py --prep-data
    """
    if prep_data:
        prep_dataset.remote(dataset=dataset, num_examples=num_examples)
        return

    # Ensure dataset is prepared
    prep_dataset.remote(dataset=dataset, num_examples=num_examples)

    # Run training
    if large:
        train_large.remote(
            model=model,
            dataset=dataset,
            num_examples=num_examples,
            epochs=epochs,
            lr=lr,
            micro_batch_size=micro_batch_size,
            max_length=max_length,
            no_lora=no_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            offload=offload,
            project_name=project_name,
            experiment_name=experiment_name,
            save_freq=save_freq,
        )
    else:
        train.remote(
            model=model,
            dataset=dataset,
            num_examples=num_examples,
            epochs=epochs,
            lr=lr,
            micro_batch_size=micro_batch_size,
            max_length=max_length,
            no_lora=no_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            offload=offload,
            project_name=project_name,
            experiment_name=experiment_name,
            save_freq=save_freq,
        )
