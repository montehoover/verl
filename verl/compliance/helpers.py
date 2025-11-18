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
import torch
from requests import HTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.metrics import jaccard_distance
from sklearn.metrics import roc_auc_score

from verl.compliance.constants import EXPLANATION_CLOSING, EXPLANATION_OPENING, LABEL_CLOSING, LABEL_OPENING, MULTIRULE_SYSTEM_PROMPT_V4, COT_OPENING, COT_CLOSING, COT_OPENING_QWEN, COT_CLOSING_QWEN, MULTIRULE_SYSTEM_PROMPT_V5, NEG_LABEL, POS_LABEL, RULES_CLOSING, RULES_OPENING, RULES_SEPARATOR
from verl.compliance.model_wrappers import VllmModelWrapper


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

# def prepare_dataset_for_verl_old(
#     dataset="tomg-group-umd/complinace",
#     subset="compliance",
#     split="train_cot",
#     num_examples=-1,
#     num_val_examples=256,
#     redownload=False,
#     local_dir="data/compliance",
# ):
#     train_path = os.path.join(local_dir, "train.parquet")
#     val_path = os.path.join(local_dir, "val.parquet")
#     if (
#         os.path.exists(train_path) and
#         os.path.exists(val_path) and 
#         not redownload
#     ):
#         print(f"Dataset already exists at {local_dir}. Use --redownload to force a new download.")
#         return train_path, val_path

#     if os.path.exists(dataset):
#         dataset = datasets.load_dataset("json", datafiles=dataset, split="train")
#     else:
#         dataset = datasets.load_dataset(dataset, subset, split=split)
    
#     if num_examples > 0 and num_examples < len(dataset):
#         dataset = dataset.shuffle(seed=42).select(range(num_examples))
    
#     train_test_split = dataset.train_test_split(test_size=num_val_examples, seed=42)
#     train_set = train_test_split['train']
#     val_set = train_test_split['test']

#     def format_to_verl(example):
#         return {"verl_stuff": {"question": example["question"], "answer": example["answer"]}}

#     train_set = train_set.map(format_to_verl)
#     val_set = val_set.map(format_to_verl)

#     train_set.to_parquet(train_path)
#     val_set.to_parquet(val_path)
#     print(f"Dataset downloaded and saved to {train_path} and {val_path}")
#     return train_path, val_path


def extract_xml_answer(text, opening_tag=LABEL_OPENING, closing_tag=LABEL_CLOSING):
    answer = text.split(opening_tag.strip())[-1]
    answer = answer.split(closing_tag.strip())[0]
    return answer.strip()


def do_preprocessing(text, model_path=None):
    """Replace <reasoning> with <thinking>"""
    # Before we were doing this just for Qwen, but now we should do it for all models.
    # if isinstance(model_path, str) and "qwen" in model_path.lower():

    if text.split()[0] == COT_OPENING:
        text = text.replace(COT_OPENING, COT_OPENING_QWEN)
        text = text.replace(COT_CLOSING, COT_CLOSING_QWEN)
    elif text.split()[0] == LABEL_OPENING:
        # The non-cot case with explanation after the answer. Replace <reasoning> with <explanation>
        text = text.replace(COT_OPENING, EXPLANATION_OPENING)
        text = text.replace(COT_CLOSING, EXPLANATION_CLOSING)

        if isinstance(model_path, str) and "qwen" in model_path.lower():
            # This is the format for the qwen chat template when enable_thinking=False.
            text = f"{COT_OPENING_QWEN}\n\n{COT_CLOSING_QWEN}\n\n{text}"
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
    do_rules_rewards=False,
    val_dataset_split=None,
    cuda_mem_test_len=None,
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
    
    if val_dataset_split is None:
        train_test_split = dataset.train_test_split(test_size=num_val_examples, seed=42)
        train_dataset = train_test_split["train"]
        val_dataset = train_test_split["test"]
    else:
        train_dataset = dataset
        val_dataset = datasets.load_dataset(dataset_path, subset, split=val_dataset_split)

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
                        "content": MULTIRULE_SYSTEM_PROMPT_V5,
                    },
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
                    "do_rules_rewards": do_rules_rewards,
                },
            }
            if cuda_mem_test_len is not None:
                data["prompt"][0]["content"] = ""
                data["prompt"][1]["content"] = " ".join(["and" for _ in range(cuda_mem_test_len)])
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn(split, model_path), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn(split.replace("train", "val")), with_indices=True)

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)
    print(f"Dataset downloaded and saved to {train_path} and {val_path}")
    num_examples = len(train_dataset)
    return train_path, val_path, num_examples


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
    api.create_repo(repo_id=hf_hub_path, private=private, exist_ok=True)
    api.upload_folder(folder_path=checkpoint_path, repo_id=hf_hub_path, repo_type="model")

def convert_to_hf(checkpoint_path, original_model):
    target_dir = f"{checkpoint_path}/hf"
    model_merger_cmd = [
        "python",
        "scripts/model_merger.py",
        "merge",
        "--backend", "fsdp",
        "--hf_model_path", original_model,
        "--local_dir", checkpoint_path,
        "--target_dir", target_dir,
        # "--hf_upload_path", hf_hub_path,
        # "--private", "True",
    ]
    subprocess.run(model_merger_cmd, check=True)
    return target_dir

def convert_and_push_to_hub(checkpoint_path, run_name, original_model=None, custom_name=None):
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

    if custom_name is not None:
        run_name = custom_name
    MAX_CHAR_LIMIT = 96  # Hugging Face Hub path limit
    compliant_run_name = run_name[:MAX_CHAR_LIMIT]  # Ensure run_name is within the limit
    hf_hub_path = f"tomg-group-umd/{compliant_run_name}"
    upload_model_to_huggingface(checkpoint_path, hf_hub_path)
    # AutoModelForCausalLM.from_pretrained(checkpoint_path).push_to_hub(hf_hub_path, private=True)
    # AutoTokenizer.from_pretrained(original_model).push_to_hub(hf_hub_path, private=True)
    
    # Cleanup
    if not hf_format_found:
        shutil.rmtree(temp_path, ignore_errors=True)
    
    return hf_hub_path

def push_to_hf_hub(checkpoint_path, run_name, original_model, raise_on_error=False, delete_checkpoint=False, custom_name=None):
    new_model_path = checkpoint_path
    try:
        new_model_path = convert_and_push_to_hub(checkpoint_path=checkpoint_path, run_name=run_name, original_model=original_model, custom_name=custom_name)
        print(f"Model pushed to Hugging Face Hub at https://huggingface.co/{new_model_path}")
    except HTTPError as e:
        print(f"There was an erro when pushing to hf hub: {e}")
        if raise_on_error:
            raise
        else:
            print("Continuing without pushing to Hugging Face Hub...")

    if delete_checkpoint:
        print(f"Deleting checkpoint at {checkpoint_path}...")
        shutil.rmtree(checkpoint_path, ignore_errors=True)
        print("Checkpoint deleted.")
    return new_model_path

def get_model_name(model_path):
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

def get_auc(model_path, dataset_path):
    # Clean up CUDA state so it doesn't create a conflit with a new VLLM instance
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # Values recommended in hf model card for non-thinking mode.
    temperature = 0.7
    top_p = 0.8
    top_k = 20
    model = VllmModelWrapper(model_path, temperature=temperature, top_k=top_k, top_p=top_p)
    dataset = datasets.load_dataset("parquet", data_files=dataset_path, split="train")

    # From data preprocessing above:
    # "prompt": [
    #     {
    #         "role": "system",
    #         "content": MULTIRULE_SYSTEM_PROMPT_V5},
    #     {
    #         "role": "user",
    #         "content": question,
    #     }
    # ],
    # "ability": "compliance",
    # "reward_model": {"style": "rule", "ground_truth": solution},
    sys_content = lambda x: x["prompt"][0]["content"]
    user_content = lambda x: x["prompt"][1]["content"]
    label = lambda x: x["reward_model"]["ground_truth"]

    messages = [model.apply_chat_template(sys_content(x), user_content(x), enable_thinking=False) for x in dataset]
    ground_truth_labels = [label(x) for x in dataset]
    y_true = [1 if label == POS_LABEL else 0 for label in ground_truth_labels]

    pos_label_probs, logit_pairs = model.get_prediction_probs(messages)
    
    auc = roc_auc_score(y_true, pos_label_probs)
    return auc


def check_pytorch_cuda_error():
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    try:
        # Try to access CUDA - will fail if memory is corrupted
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            try:
                # Try basic CUDA operations
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                
                # Check memory usage
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                max_allocated = torch.cuda.max_memory_allocated(i)
                
                # If we're at 100% memory usage, that's likely the issue
                total_memory = torch.cuda.get_device_properties(i).total_memory
                if reserved / total_memory > 0.95:
                    return f"GPU {i} memory exhausted: {reserved/1e9:.1f}GB / {total_memory/1e9:.1f}GB"
                    
            except RuntimeError as cuda_err:
                if "out of memory" in str(cuda_err).lower():
                    return f"CUDA OOM on GPU {i}: {cuda_err}"
                elif "cuda" in str(cuda_err).lower():
                    return f"CUDA error on GPU {i}: {cuda_err}"
                    
    except Exception as e:
        return f"PyTorch CUDA check failed: {e}"
    
    return None

####################
# Reward functions
####################
# Reward between 0.0 and 1.0.
# Breakdown:
#   Label Correctness: 0.40
#   Rule Correctness: 0.30
#   Format: 0.30
#     Labels printed correctly: 0.06 (PASS/FAIL)
#     All 4 xml tag plus newlines exactly: 0.06
#     All 4 xml tags in the right order: 0.06
#     Xml tags present at all: 0.02 per tag for total of 0.12
def compute_reward(data_source, solution_str, ground_truth, extra_info=None):
    assert "compliance" in data_source, f"Data source {data_source} is not a compliance dataset. Expected tomg-group-umd/compliance or montehoover/compliance."
    assert extra_info is not None, "Extra info must be provided to compute the reward for rules_violated. Check the data preprocessing step to make sure there is a column in the dataset named extra_info."

    pos_label = POS_LABEL
    neg_label = NEG_LABEL
    label_opening = LABEL_OPENING
    label_closing = LABEL_CLOSING
    rules_opening = RULES_OPENING
    rules_closing = RULES_CLOSING

    # We used to have a check for Qwen models here, but we should use the same tags for all models now.
    # model = extra_info.get("model", None) if extra_info else None
    # if model and "qwen" in model.lower():

    cot_opening = COT_OPENING_QWEN
    cot_closing = COT_CLOSING_QWEN
    
    # else:
    #     cot_opening = COT_OPENING
    #     cot_closing = COT_CLOSING

    ground_truth_label = ground_truth
    full_ground_truth = extra_info.get("answer", None)
    do_rules_rewards = extra_info.get("do_rules_rewards", False)

    if do_rules_rewards:
        rules_reward = rules_reward_func(solution_str, full_ground_truth, rules_opening, rules_closing, points=0.30)
        correctness_reward = correctness_reward_func(solution_str, ground_truth_label, label_opening, label_closing, points=0.40)
        label_format_reward = label_reward_func(solution_str, label_opening, label_closing, pos_label, neg_label, points=0.06)
        strict_format_reward = strict_format_reward_func(solution_str, cot_opening, cot_closing, label_opening, label_closing, points=0.06)
        soft_format_reward = soft_format_reward_func(solution_str, cot_opening, cot_closing, label_opening, label_closing, points=0.06)
        xml_count_reward = xmlcount_reward_func(solution_str, cot_opening, cot_closing, label_opening, label_closing, rules_opening, rules_closing, points=0.02)
        return rules_reward + correctness_reward + label_format_reward + strict_format_reward + soft_format_reward + xml_count_reward
    else:
        correctness_reward = correctness_reward_func(solution_str, ground_truth_label, label_opening, label_closing, points=0.68)
        label_format_reward = label_reward_func(solution_str, label_opening, label_closing, pos_label, neg_label, points=0.08)
        strict_format_reward = strict_format_reward_func(solution_str, cot_opening, cot_closing, label_opening, label_closing, points=0.08)
        soft_format_reward = soft_format_reward_func(solution_str, cot_opening, cot_closing, label_opening, label_closing, points=0.08)
        xml_count_reward = xmlcount_reward_func(solution_str, cot_opening, cot_closing, label_opening, label_closing, rules_opening, rules_closing, points=0.02)
        return correctness_reward + label_format_reward + strict_format_reward + soft_format_reward + xml_count_reward

def correctness_reward_func(model_output, ground_truth, label_opening, label_closing, points=0.40):
    prediction = extract_xml_answer(model_output, label_opening, label_closing)
    return points if prediction == ground_truth else 0.0 

def rules_reward_func(model_output, full_ground_truth, rules_opening, rules_closing, points=0.30):
    ground_truth_rules_string = extract_xml_answer(full_ground_truth, rules_opening, rules_closing)
    predicted_rules_string = extract_xml_answer(model_output, rules_opening, rules_closing)
    ground_truth_rules = set([s.strip() for s in ground_truth_rules_string.split(RULES_SEPARATOR)])
    predicted_rules = set([s.strip() for s in predicted_rules_string.split(RULES_SEPARATOR)])
    score = 1 - jaccard_distance(ground_truth_rules, predicted_rules) # jaccard dist is between 0 and 1, so for the score 1.0 means perfect match, 0.0 means no overlap
    return score * points

def label_reward_func(model_output, label_opening, label_closing, pos_label, neg_label, points=0.06):
    """Reward function that checks if the label is exactly PASS or FAIL."""
    prediction = extract_xml_answer(model_output, label_opening, label_closing)
    return points if prediction in [pos_label, neg_label] else 0.0

def strict_format_reward_func(model_output, cot_opening, cot_closing, label_opening, label_closing, points=0.06):
    """Reward function that checks if the completion is in XML_COT_FORMAT, strictly adhering to newlines before and after every tag."""
    # pattern = r"^\n<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    pattern = fr"^\n{cot_opening}\n.*?\n{cot_closing}\n{label_opening}\n.*?\n{label_closing}\n$"
    match = re.search(pattern, model_output)
    return points if match else 0.0

def soft_format_reward_func(model_output, cot_opening, cot_closing, label_opening, label_closing, points=0.06):
    """Reward function that checks if the completion is in XML_COT_FORMAT, with flexibility in newlines and whitespace."""
    # pattern = r"^\s*<reasoning>\s*.*?\s*</reasoning>\s*<answer>\s*.*?\s*</answer>\s*$"
    pattern = fr"^\s*{cot_opening}\s*.*?\s*{cot_closing}\s*{label_opening}\s*.*?\s*{label_closing}\s*$"
    match = re.search(pattern, model_output)
    return points if match else 0.0

def xmlcount_reward_func(model_output, cot_opening, cot_closing, label_opening, label_closing, rules_opening=None, rules_closing=None, points=0.02):
    """We want to encourage xml tags to be present, so just give rewards if they are present at all. Let other functions handle extraneous stuff."""
    reward = 0.0
    if model_output.count(cot_opening) == 1:
        reward += points
    if model_output.count(cot_closing) == 1:
        reward += points
    if model_output.count(label_opening) == 1:
        reward += points
    if model_output.count(label_closing) == 1:
        reward += points
    if rules_opening is not None and rules_closing is not None:
        if model_output.count(rules_opening) == 1:
            reward += points
        if model_output.count(rules_closing) == 1:
            reward += points
    return reward
#######################
# End reward functions
#######################

