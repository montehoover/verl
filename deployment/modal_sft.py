# ---
# cmd: ["modal", "run", "deployment/modal_sft.py::train"]
# ---

# # Train a model using Supervised Fine-Tuning (SFT) with verl

# This example demonstrates how to perform SFT on Modal using the [verl](https://github.com/volcengine/verl) framework.
# SFT is the initial supervised training phase before reinforcement learning, where the model learns from question-answer pairs.

# ## Setup

import subprocess
from pathlib import Path

import modal

# ## Defining the image and app

app = modal.App("verl-sft")

# We define an image where we clone the verl repo and install its dependencies.

VERL_REPO_PATH: Path = Path("/root/verl")
image = (
    modal.Image.from_registry("verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1")
    .apt_install("git")
    .run_commands(f"git clone https://github.com/volcengine/verl {VERL_REPO_PATH}")
    .uv_pip_install("verl[vllm]==0.4.1")
)

# ## Defining the dataset

# We use the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset of math problems
# and a [Modal Volume](https://modal.com/docs/guide/volumes#volumes) to store the data.

DATA_PATH: Path = Path("/data")
data_volume: modal.Volume = modal.Volume.from_name("verl-sft-data", create_if_missing=True)


# We write a Modal Function to populate the Volume with the data.
@app.function(image=image, volumes={DATA_PATH: data_volume})
def prep_dataset() -> None:
    subprocess.run(
        [
            "python",
            VERL_REPO_PATH / "examples" / "data_preprocess" / "gsm8k.py",
            "--local_dir",
            DATA_PATH,
        ],
        check=True,
    )


# You can kick off the dataset download with
# `modal run deployment/modal_sft.py::prep_dataset`

# ## Kicking off a training run

MODELS_PATH: Path = Path("/models")
MINUTES: int = 60

# We define a Volume for storing model checkpoints.
checkpoints_volume: modal.Volume = modal.Volume.from_name("verl-sft-checkpoints", create_if_missing=True)

# Now, we write a Modal Function for kicking off the SFT training run.
# If you wish to use Weights & Biases, you'll need to create a Weights & Biases [Secret.](https://modal.com/docs/guide/secrets#secrets)


@app.function(
    image=image,
    gpu="A100:2",
    volumes={
        MODELS_PATH: checkpoints_volume,
        DATA_PATH: data_volume,
    },
    # secrets=[modal.Secret.from_name("wandb-secret")],  # Uncomment if using WandB
    timeout=24 * 60 * MINUTES,
)
def train(*arglist) -> None:
    data_volume.reload()

    cmd: list[str] = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "-m",
        "verl.trainer.fsdp_sft_trainer",
        f"data.train_files={DATA_PATH / 'train.parquet'}",
        f"data.val_files={DATA_PATH / 'test.parquet'}",
        "data.prompt_key=extra_info",
        "data.response_key=extra_info",
        "optim.lr=1e-4",
        "data.prompt_dict_keys=['question']",
        "+data.response_dict_keys=['answer']",
        "data.micro_batch_size_per_gpu=4",
        "model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct",
        f"trainer.default_local_dir={MODELS_PATH}",
        "trainer.project_name=verl-sft-gsm8k",
        "trainer.experiment_name=qwen2.5-0.5b-sft",
        "trainer.logger=['console']",
        "trainer.total_epochs=50",
        # LoRA configuration
        "model.lora_rank=32",
        "model.lora_alpha=16",
        "model.target_modules=all-linear",
    ]
    if arglist:
        cmd.extend(arglist)

    subprocess.run(cmd, check=True)
    checkpoints_volume.commit()


# You can now run the training using `modal run --detach deployment/modal_sft.py::train`,
# or pass in additional args like `modal run --detach deployment/modal_sft.py::train -- trainer.total_epochs=100`
