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

def prepare_dataset_for_verl(
    dataset="tomg-group-umd/complinace",
    subset="compliance",
    split="train_cot",
    num_examples=-1,
    val_examples=256,
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
    
    train_test_split = dataset.train_test_split(test_size=val_examples, seed=42)
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
