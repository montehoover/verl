import os
import re


def get_last_checkpoint_path(run_name, checkpoint_dir="checkpoints"):
    checkpoint_folders = [
        folder for folder in os.listdir(f"{checkpoint_dir}/{run_name}")
        if os.path.isdir(os.path.join(f"{checkpoint_dir}/{run_name}", folder))
    ]
    checkpoint_numbers = [int(folder.split("_")[-1]) for folder in checkpoint_folders]
    last_checkpoint = f"global_step_{max(checkpoint_numbers)}"
    checkpoint_path = f"{checkpoint_dir}/{run_name}/{last_checkpoint}"
    
    # See if the checkpoint is in hf format
    for file in os.listdir(checkpoint_path):
        if file.endswith(".safetensors"):
            return checkpoint_path
    
    # If a hf model file isn't found, look for a folder called "actor"
    for file in os.listdir(checkpoint_path):
        if os.path.isdir(os.path.join(checkpoint_path, file)) and file == "actor":
            checkpoint_path = os.path.join(checkpoint_path, file)
            return checkpoint_path
     
    print(f"Checkpoint path {checkpoint_path} does not contain a valid checkpoint. Returning the path for investigation.")
    return checkpoint_path


LORA_TARGET_MODULE_CHOICES = ["all-linear", "all-linear-and-embedding", "all-attention", "qv-only"]


def get_lora_target_modules(name):
    """Convert a friendly LoRA target module name to the Hydra config value.

    Passed to peft's LoraConfig via actor_rollout_ref.model.target_modules (GRPO/PPO)
    or model.target_modules (SFT). Module names are for Qwen-style architectures.
    """
    mapping = {
        "all-linear": "all-linear",
        "all-linear-and-embedding": "[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,embed_tokens]",
        "all-attention": "[q_proj,k_proj,v_proj,o_proj]",
        "qv-only": "[q_proj,v_proj]",
    }
    if name not in mapping:
        raise ValueError(f"Unknown lora_target_modules: {name!r}. Choose from: {list(mapping.keys())}")
    return mapping[name]


def get_short_model_name(model_path):
    """
    Get short model name from model path.
    """
    # The leading .* gobbles up as much as possible, so if there are multiple
    # instances of Qwen, it only returns the last one instead of both.
    # The .*? is a non-greedy match, so it will stop at the first instance of B.
    patterns = [
        r'.*(Qwen.*?B(?:-Base)?)', # :? makes the parenthesis non-capturing, so it allows a match without create a separate group. The contents are still captured by the outer group.
        r'.*(Llama.*?B)',
        r'.*(llama.*?b)',
        r'.*(wildguard)',
    ]
    for patttern in patterns:
        m = re.search(patttern, model_path)
        if m:
            return m.group(1)               # group(0) is the whole match
        
    # Just send back the whole path as the model name if we can't find a match.
    # Replace slashes with underscores to make it a valid model name.
    return model_path.replace("/", "_")
