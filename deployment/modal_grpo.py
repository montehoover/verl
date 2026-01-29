# ---
# cmd: ["modal", "run", "deployment/modal_grpo.py::train"]
# ---
import re
import subprocess
from pathlib import Path
from typing import Literal, Optional

import modal

# ## Setup

app = modal.App("verl-grpo")

VERL_REPO_PATH: Path = Path("/root/verl")
VERL_FORK_URL: str = "https://github.com/montehoover/verl.git"
VERL_BRANCH: str = "dynaguard1"

image = (
    modal.Image.from_registry("verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1")
    .apt_install("git")
    .run_commands(f"git clone --branch {VERL_BRANCH} {VERL_FORK_URL} {VERL_REPO_PATH}")
    .uv_pip_install("verl[vllm]==0.4.1")
)

# ## Dataset setup

DATA_PATH: Path = Path("/data")
data_volume: modal.Volume = modal.Volume.from_name("verl-grpo-data", create_if_missing=True)


@app.function(image=image, volumes={DATA_PATH: data_volume})
def prep_dataset() -> None:
    """Download and prepare GSM8K dataset"""
    subprocess.run(
        [
            "python",
            VERL_REPO_PATH / "examples" / "data_preprocess" / "gsm8k.py",
            "--local_dir",
            DATA_PATH,
        ],
        check=True,
    )
    data_volume.commit()


# Run with: modal run deployment/modal_grpo.py::prep_dataset

# ## Reward function for GSM8K


def extract_solution(solution_str: str, method: Literal["strict", "flexible"] = "strict") -> Optional[str]:
    """Extract the numerical answer from the solution string"""
    assert method in ["strict", "flexible"]

    if method == "strict":
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer: Optional[str] = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer: Optional[str] = None
        if len(answer) == 0:
            pass
        else:
            invalid_str: list[str] = ["", "."]
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_reward(data_source: str, solution_str: str, ground_truth: str, extra_info: dict) -> float:
    """Compute reward: 1.0 for correct answer, 0.0 otherwise"""
    answer = extract_solution(solution_str=solution_str, method="strict")
    if answer is None:
        return 0.0
    else:
        if answer == ground_truth:
            return 1.0
        else:
            return 0.0


PATH_TO_REWARD_FUNCTION: Path = Path("/root/modal_grpo.py")
REWARD_FUNCTION_NAME: str = "compute_reward"

# ## Training

MODELS_PATH: Path = Path("/models")
MINUTES: int = 60

checkpoints_volume: modal.Volume = modal.Volume.from_name("verl-grpo-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100:2",
    volumes={
        MODELS_PATH: checkpoints_volume,
        DATA_PATH: data_volume,
    },
    timeout=24 * 60 * MINUTES,
)
def train(*arglist) -> None:
    """Run GRPO training on GSM8K dataset"""
    data_volume.reload()

    cmd: list[str] = _build_train_cmd(use_wandb=False)

    if arglist:
        cmd.extend(arglist)

    subprocess.run(cmd, check=True)
    checkpoints_volume.commit()


def _build_train_cmd(*, use_wandb: bool) -> list[str]:
    logger = "trainer.logger=['console','wandb']" if use_wandb else "trainer.logger=console"
    return [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={DATA_PATH / 'train.parquet'}",
        f"data.val_files={DATA_PATH / 'test.parquet'}",
        "data.train_batch_size=256",
        "data.max_prompt_length=512",
        "data.max_response_length=512",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        logger,
        "trainer.project_name=verl-grpo-gsm8k",
        "trainer.experiment_name=qwen2.5-0.5b-grpo",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        "trainer.save_freq=10",
        "trainer.test_freq=10",
        f"trainer.default_local_dir={MODELS_PATH}",
        "trainer.resume_mode=auto",
        "trainer.total_epochs=50",
        f"custom_reward_function.path={str(PATH_TO_REWARD_FUNCTION)}",
        f"custom_reward_function.name={REWARD_FUNCTION_NAME}",
    ]

# Run with: modal run --detach deployment/modal_grpo.py::train
# Or with custom args: modal run --detach deployment/modal_grpo.py::train -- trainer.total_epochs=100


@app.function(
    image=image,
    gpu="H100:2",
    volumes={
        MODELS_PATH: checkpoints_volume,
        DATA_PATH: data_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 60 * MINUTES,
)
def train_wandb(*arglist) -> None:
    """Run GRPO training with wandb logging (requires wandb-secret)"""
    data_volume.reload()

    cmd: list[str] = _build_train_cmd(use_wandb=True)

    if arglist:
        cmd.extend(arglist)

    subprocess.run(cmd, check=True)
    checkpoints_volume.commit()


# Run with: modal run --detach deployment/modal_grpo.py::train_wandb

# ## Inference

VLLM_PORT: int = 8000


def get_latest_checkpoint_file_path():
    """Get path to the latest model checkpoint"""
    with open(MODELS_PATH / "latest_checkpointed_iteration.txt") as f:
        latest_checkpoint_index = int(f.read())
    return str(MODELS_PATH / f"global_step_{latest_checkpoint_index}" / "actor" / "huggingface")


vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "vllm==0.9.1",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .env({"VLLM_USE_V1": "1"})
)

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="H100:2",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={"/root/.cache/vllm": vllm_cache_vol, MODELS_PATH: checkpoints_volume},
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """Serve the trained model using vLLM"""
    import subprocess

    latest_checkpoint_file_path = get_latest_checkpoint_file_path()

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        latest_checkpoint_file_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        "2",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)


# Deploy with: modal deploy deployment/modal_grpo.py
