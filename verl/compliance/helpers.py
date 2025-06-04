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
import shutil
from requests import HTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.compliance.constants import EXPLANATION_CLOSING, EXPLANATION_OPENING, LABEL_CLOSING, LABEL_OPENING, MULTIRULE_SYSTEM_PROMPT_V4, COT_OPENING, COT_CLOSING, COT_OPENING_QWEN, COT_CLOSING_QWEN, MULTIRULE_SYSTEM_PROMPT_V5, NEG_LABEL, POS_LABEL


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


def do_preprocessing(text, model_path=None):
    if isinstance(model_path, str) and "qwen" in model_path.lower():
        if text.split()[0] == COT_OPENING:
            text = text.replace(COT_OPENING, COT_OPENING_QWEN)
            text = text.replace(COT_CLOSING, COT_CLOSING_QWEN)
        elif text.split()[0] == LABEL_OPENING:
            text = text.replace(COT_OPENING, EXPLANATION_OPENING)
            text = text.replace(COT_CLOSING, EXPLANATION_CLOSING)
    return text


def prepare_dataset_for_verl(
    dataset_path="tomg-group-umd/complinace",
    subset="compliance",
    split="train_32000_mix",
    num_examples=-1,
    num_val_examples=256,
    redownload=False,
    local_dir="data/compliance",
    model_path="",
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
    def make_map_fn(split, model_path=None):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            answer_raw = example.pop("answer")

            question = do_preprocessing(question_raw, model_path=model_path)
            answer = do_preprocessing(answer_raw, model_path=model_path)

            solution = extract_xml_answer(answer_raw)
            data = {
                "data_source": dataset_path,
                "prompt": [
                    {
                        "role": "system",
                        "content": MULTIRULE_SYSTEM_PROMPT_V5},
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "compliance",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "model": model_path,
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                },
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn(split, model_path), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn(split.replace("train", "val")), with_indices=True)

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)
    print(f"Dataset downloaded and saved to {train_path} and {val_path}")
    return train_path, val_path


def get_subset(train_path, num_examples, filetype="parquet"):
    train_dataset = datasets.load_dataset(filetype, data_files=train_path, split="train")
    if num_examples <= 0 or num_examples > len(train_dataset):
        print(f"Requested subset size of {num_examples} but the original dataset only has {len(train_dataset)} examples. Returning the full dataset.")
        return train_path
    subset_dataset = train_dataset.select(range(num_examples))
    subset_path = train_path.replace(f".{filetype}", f"_{num_examples}.{filetype}")
    subset_dataset.to_parquet(subset_path)
    return subset_path


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

def upload_model_to_huggingface(checkpoint_path, hf_hub_path, private=True):
    from huggingface_hub import HfApi
    api = HfApi()
    hf_hub_path = hf_hub_path[:96] # Ensure the path is not too long, as HF has a limit of 96 characters.
    api.create_repo(repo_id=hf_hub_path, private=private, exist_ok=True)
    api.upload_folder(folder_path=checkpoint_path, repo_id=hf_hub_path, repo_type="model")

def convert_and_push_to_hub(checkpoint_path, run_name, original_model=None):
    assert os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path), f"We need the path to a directory that contains model files, not a file. Got {checkpoint_path} instead."
    
    hf_format_found = False
    for file in os.listdir(checkpoint_path):
        if file.endswith(".safetensors"):
            hf_format_found = True
            break
    
    if not hf_format_found:
        assert original_model is not None, f"Model name must be provided if no .safetensors file is found."
        temp_path = f"temp/{time.time_ns()}"
        model_merger_cmd = [
            "python",
            "scripts/model_merger.py",
            "merge",
            "--backend", "fsdp",
            "--hf_model_path", original_model,
            "--local_dir", checkpoint_path,
            "--target_dir", temp_path,
            # "--hf_upload_path", hf_hub_path,
            # "--private", "True",
        ]
        subprocess.run(model_merger_cmd, check=True)
        checkpoint_path = temp_path

    hf_hub_path = f"tomg-group-umd/c_{run_name}"
    upload_model_to_huggingface(checkpoint_path, hf_hub_path)
    # AutoModelForCausalLM.from_pretrained(checkpoint_path).push_to_hub(hf_hub_path, private=True)
    # AutoTokenizer.from_pretrained(original_model).push_to_hub(hf_hub_path, private=True)
    
    # Cleanup
    if not hf_format_found:
        shutil.rmtree(temp_path, ignore_errors=True)
    
    return hf_hub_path

def push_to_hf_hub(checkpoint_path, run_name, original_model, raise_on_error=False):
    new_model_path = checkpoint_path
    try:
        new_model_path = convert_and_push_to_hub(checkpoint_path=checkpoint_path, run_name=run_name, original_model=original_model)
        print(f"Model pushed to Hugging Face Hub at {new_model_path}")
    except HTTPError as e:
        print(f"There was an erro when pushing to hf hub: {e}")
        if raise_on_error:
            raise
        else:
            print("Continuing without pushing to Hugging Face Hub...")
    return new_model_path

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
    pos_label = POS_LABEL
    neg_label = NEG_LABEL
    label_opening = LABEL_OPENING
    label_closing = LABEL_CLOSING
    cot_opening = COT_OPENING
    cot_closing = COT_CLOSING
    model = extra_info.get("model", None) if extra_info else None
    if model and "qwen" in model.lower():
        cot_opening = COT_OPENING_QWEN
        cot_closing = COT_CLOSING_QWEN

    assert "compliance" in data_source, f"Data source {data_source} is not a compliance dataset. Expected tomg-group-umd/compliance or montehoover/compliance."
    correctness_reward = correctness_reward_func(solution_str, ground_truth, label_opening, label_closing)
    label_format_reward = label_reward_func(solution_str, label_opening, label_closing, pos_label, neg_label)
    strict_format_reward = strict_format_reward_func(solution_str, cot_opening, cot_closing, label_opening, label_closing)
    soft_format_reward = soft_format_reward_func(solution_str, cot_opening, cot_closing, label_opening, label_closing)
    xml_count_reward = xmlcount_reward_func(solution_str, cot_opening, cot_closing, label_opening, label_closing)
    return correctness_reward + label_format_reward + strict_format_reward + soft_format_reward + xml_count_reward

def correctness_reward_func(model_output, ground_truth, label_opening, label_closing):
    prediction = extract_xml_answer(model_output, label_opening, label_closing)
    return 0.68 if prediction == ground_truth else 0.0 

def label_reward_func(model_output, label_opening, label_closing, pos_label, neg_label):
    """Reward function that checks if the label is exactly PASS or FAIL."""
    prediction = extract_xml_answer(model_output, label_opening, label_closing)
    return 0.08 if prediction in [pos_label, neg_label] else 0.0

def strict_format_reward_func(model_output, cot_opening, cot_closing, label_opening, label_closing):
    """Reward function that checks if the completion is in XML_COT_FORMAT, strictly adhering to newlines before and after every tag."""
    # pattern = r"^\n<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    pattern = fr"^\n{cot_opening}\n.*?\n{cot_closing}\n{label_opening}\n.*?\n{label_closing}\n$"
    match = re.search(pattern, model_output)
    return 0.08 if match else 0.0

def soft_format_reward_func(model_output, cot_opening, cot_closing, label_opening, label_closing):
    """Reward function that checks if the completion is in XML_COT_FORMAT, with flexibility in newlines and whitespace."""
    # pattern = r"^\s*<reasoning>\s*.*?\s*</reasoning>\s*<answer>\s*.*?\s*</answer>\s*$"
    pattern = fr"^\s*{cot_opening}\s*.*?\s*{cot_closing}\s*{label_opening}\s*.*?\s*{label_closing}\s*$"
    match = re.search(pattern, model_output)
    return 0.08 if match else 0.0

def xmlcount_reward_func(model_output, cot_opening, cot_closing, label_opening, label_closing):
    """We want to encourage xml tags to be present, so just give rewards if they are present at all. Let other functions handle extraneous stuff."""
    count = 0.0
    # if model_output.count("<reasoning>\n") == 1:
    #     count += 0.02
    # if model_output.count("\n</reasoning>\n") == 1:
    #     count += 0.02
    # if model_output.count("\n<answer>\n") == 1:
    #     count += 0.02
    # if model_output.count("\n</answer>") == 1:
    #     count += 0.02
    if model_output.count(f"{cot_opening}\n") == 1:
        count += 0.02
    if model_output.count(f"\n{cot_closing}\n") == 1:
        count += 0.02
    if model_output.count(f"\n{label_opening}\n") == 1:
        count += 0.02
    if model_output.count(f"\n{label_closing}") == 1:
        count += 0.02
    return count
#######################
# End reward functions
#######################

