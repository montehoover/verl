"""
Download the dataset from huggingface, shuffle, and create a val split.
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil

from verl.compliance.constants import LABEL_CLOSING, LABEL_OPENING, MULTIRULE_SYSTEM_PROMPT_V4


def configure_logging(log_level=None, ext_level_bump=1, log_file=f"{time.time_ns()}.log"):
    if log_file:
        # Create a new log file or append to an existing one
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as f:
            pass

    # Create a custom level that is between INFO and WARNING
    logging.addLevelName(25, "NOTICE")
    notice = lambda self, message, *args, **kwargs: self._log(25, message, args, **kwargs)
    logging.Logger.notice = notice

    # Determine log level: CLI argument > Environment variable > Default (NOTICE)
    log_level = (log_level or os.getenv("LOG_LEVEL", "NOTICE")).upper()
    logging.basicConfig(
        level=log_level,
        format="{name}:{levelname}: {message}",
        style="{",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def prepare_dataset_for_verl_old(
    dataset="tomg-group-umd/complinace",
    subset="compliance",
    split="train_cot",
    num_examples=-1,
    num_val_examples=256,
    redownload=False,
    local_dir="data/compliance",
):
    train_path = os.path.join(local_dir, "train.parquet")
    val_path = os.path.join(local_dir, "val.parquet")
    if (
        os.path.exists(train_path) and
        os.path.exists(val_path) and 
        not redownload
    ):
        print(f"Dataset already exists at {local_dir}. Use --redownload to force a new download.")
        return train_path, val_path

    if os.path.exists(dataset):
        dataset = datasets.load_dataset("json", datafiles=dataset, split="train")
    else:
        dataset = datasets.load_dataset(dataset, subset, split=split)
    
    if num_examples > 0 and num_examples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(num_examples))
    
    train_test_split = dataset.train_test_split(test_size=num_val_examples, seed=42)
    train_set = train_test_split['train']
    val_set = train_test_split['test']

    def format_to_verl(example):
        return {"verl_stuff": {"question": example["question"], "answer": example["answer"]}}

    train_set = train_set.map(format_to_verl)
    val_set = val_set.map(format_to_verl)

    train_set.to_parquet(train_path)
    val_set.to_parquet(val_path)
    print(f"Dataset downloaded and saved to {train_path} and {val_path}")
    return train_path, val_path


def extract_xml_answer(text, opening_tag=LABEL_OPENING, closing_tag=LABEL_CLOSING):
    answer = text.split(opening_tag.strip())[-1]
    answer = answer.split(closing_tag.strip())[0]
    return answer.strip()


def prepare_dataset_for_verl(
    dataset_path="tomg-group-umd/complinace",
    subset="compliance",
    split="train_32000_mix",
    num_examples=-1,
    num_val_examples=256,
    redownload=False,
    local_dir="data/compliance",
):
    train_path = os.path.join(local_dir, "train.parquet")
    val_path = os.path.join(local_dir, "val.parquet")
    if (
        os.path.exists(train_path) and
        os.path.exists(val_path) and 
        not redownload
    ):
        print(f"Dataset already exists at {local_dir}. Use --redownload to force a new download.")
        return train_path, val_path

    if os.path.exists(dataset_path):
        dataset = datasets.load_dataset("json", datafiles=dataset_path, split="train")
    else:
        dataset = datasets.load_dataset(dataset_path, subset, split=split)
    
    if num_examples > 0 and num_examples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(num_examples))
    
    train_test_split = dataset.train_test_split(test_size=num_val_examples, seed=42)
    train_dataset = train_test_split["train"]
    val_dataset = train_test_split["test"]

    # Taken from examples/data_preprocess/gsm8k.py
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            answer_raw = example.pop("answer")

            solution = extract_xml_answer(answer_raw)
            data = {
                "data_source": dataset_path,
                "prompt": [
                    {
                        "role": "system",
                        "content": MULTIRULE_SYSTEM_PROMPT_V4},
                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ],
                "ability": "compliance",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn(split), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn(split.replace("train", "val")), with_indices=True)

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)
    print(f"Dataset downloaded and saved to {train_path} and {val_path}")
    return train_path, val_path


def run_subprocess(cmd, logger, check=True):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # stream stdout
    for line in process.stdout:
        logger.info(line.rstrip())
    # stream stderr
    for line in process.stderr:
        logger.error(line.rstrip())
    returncode = process.wait()
    if check:
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
    return returncode


def get_last_checkpoint_path(run_name):
    checkpoint_folders = [
        folder for folder in os.listdir(f"checkpoints/{run_name}")
        if os.path.isdir(os.path.join(f"checkpoints/{run_name}", folder))
    ]
    checkpoint_numbers = [int(folder.split("_")[-1]) for folder in checkpoint_folders]
    last_checkpoint = f"global_step_{max(checkpoint_numbers)}"
    checkpoint_path = f"checkpoints/{run_name}/{last_checkpoint}"
    return checkpoint_path


def push_to_hub(checkpoint_path, run_name, model_name=None):
    assert os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path), f"We need the path to a directory that contains model files, not a file. Got {checkpoint_path} instead."
    
    hf_format_found = False
    for file in os.listdir(checkpoint_path):
        if file.endswith(".safetensors"):
            hf_format_found = True
            break
    
    if not hf_format_found:
        assert model_name is not None, f"Model name must be provided if no .safetensors file is found."
        temp_path = f"temp/{time.time_ns()}"
        model_merger_cmd = [
            "python",
            "scripts/model_merger.py",
            "--backend", "fsdp",
            "--hf_model_path", model_name,
            "--local_dir", checkpoint_path,
            "--target_dir", temp_path,
        ]
        subprocess.run(model_merger_cmd, check=True)
        checkpoint_path = temp_path

    hf_hub_path = f"tomg-group-umd/compliance_{run_name}"
    AutoModelForCausalLM.from_pretrained(checkpoint_path).push_to_hub(hf_hub_path, private=True)
    
    # Cleanup
    if not hf_format_found:
        shutil.rmtree(temp_path, ignore_errors=True)
    
    return hf_hub_path


def get_model_name(model_path):
    # The leading .* gobbles up as much as possible, so if there are multiple
    # instances of Qwen, it only returns the last one instead of both.
    # The .*? is a non-greedy match, so it will stop at the first instance of B.
    patterns = [
        r'.*(Qwen.*?B)',
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


####################
# Reward functions
####################
# Reward between 0.0 and 1.0.
# Breakdown:
#   Correctness: 0.68
#   Format: 0.32
#     Labels printed correctly: 0.08 (PASS/FAIL)
#     All 4 xml tag plus newlines exactly: 0.08
#     All 4 xml tags in the right order: 0.08
#     Xml tags present at all: 0.02 per tag for total of 0.08
def compute_reward(data_source, solution_str, ground_truth, extra_info=None):
    assert "compliance" in data_source, f"Data source {data_source} is not a compliance dataset. Expected tomg-group-umd/compliance or montehoover/compliance."
    correctness_reward = correctness_reward_func(solution_str, ground_truth)
    label_format_reward = label_reward_func(solution_str)
    strict_format_reward = strict_format_reward_func(solution_str)
    soft_format_reward = soft_format_reward_func(solution_str)
    xml_count_reward = xmlcount_reward_func(solution_str)
    return correctness_reward + label_format_reward + strict_format_reward + soft_format_reward + xml_count_reward

def correctness_reward_func(model_output, ground_truth):
    prediction = extract_xml_answer(model_output)
    return 0.68 if prediction == ground_truth else 0.0 

def label_reward_func(model_output):
    """Reward function that checks if the label is exactly PASS or FAIL."""
    prediction = extract_xml_answer(model_output)
    return 0.08 if prediction in ["PASS", "FAIL"] else 0.0

def strict_format_reward_func(model_output):
    """Reward function that checks if the completion is in XML_COT_FORMAT, strictly adhering to newlines before and after every tag."""
    pattern = r"^\n<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    match = re.search(pattern, model_output)
    return 0.08 if match else 0.0

def soft_format_reward_func(model_output):
    """Reward function that checks if the completion is in XML_COT_FORMAT, with flexibility in newlines and whitespace."""
    pattern = r"^\s*<reasoning>\s*.*?\s*</reasoning>\s*<answer>\s*.*?\s*</answer>\s*$"
    match = re.search(pattern, model_output)
    return 0.08 if match else 0.0

def xmlcount_reward_func(model_output) -> float:
    """We want to encourage xml tags to be present, so just give rewards if they are present at all. Let other functions handle extraneous stuff."""
    count = 0.0
    if model_output.count("<reasoning>\n") == 1:
        count += 0.02
    if model_output.count("\n</reasoning>\n") == 1:
        count += 0.02
    if model_output.count("\n<answer>\n") == 1:
        count += 0.02
    if model_output.count("\n</answer>") == 1:
        count += 0.02
    return count
#######################
# End reward functions
#######################

