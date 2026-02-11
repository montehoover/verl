import os
import re
import datasets


def extract_solution_gsm8k(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def resolve_dataset_subset(dataset_path, dataset_subset):
    """Return a config name or None to pass as name= in load_dataset.

    Passing None defers to HuggingFace defaults; it loads single-config datasets
    and raises if the dataset requires an explicit config name.
    """
    config_names = datasets.get_dataset_config_names(dataset_path)
    if not config_names:
        return None
    else:
        print(f"Note: No subset provided. Using first subset/config name: {config_names[0]}")
        return config_names[0]


def load_hf_dataset(dataset_path, dataset_subset):
    resolved_subset = dataset_subset or resolve_dataset_subset(dataset_path, dataset_subset)
    return datasets.load_dataset(dataset_path, name=resolved_subset)


def preprocess_dataset_gsm8k(
    hf_dataset_name="openai/gsm8k",
    hf_dataset_subset=None,
    local_dataset_path=None,
    local_save_dir="data/gsm8k",
    num_examples=-1,
    val_split=None,
    val_size=0.0,
):
    """Download and preprocess the GSM8K dataset into verl's expected parquet format.

    Transforms each row into the schema verl expects, and saves train/test splits as
    parquet files. The output parquet files contain the following fields per row:

        - "data_source": The dataset identifier string (e.g. "openai/gsm8k"). Used by
            the reward manager to route to the correct reward function.
        - "prompt": A list of chat messages in HuggingFace chat_template format
            (e.g. [{"role": "user", "content": "..."}]). The tokenizer in RLHFDataset
            applies the chat template and tokenizes this.
        - "ability": Task category string (set to "math" for GSM8K).
        - "reward_model": A dict with "style" ("rule") and "ground_truth" (the
            extracted numeric answer as a string). The ground_truth is what gets
            passed to the reward function during training.
        - "extra_info": A dict with metadata ("split", "index", "answer", "question").
            Passed through to the reward function as extra_info.

    The output format is designed to pair with the verl reward function format.
    See verl/tomlab/reward_functions.py, where "reward_model.ground_truth" becomes the
    ground_truth argument and "data_source" becomes the data_source argument.

    Args:
        hf_dataset_name: HuggingFace dataset identifier. Stored as "data_source" in
            each row so the reward manager can identify which reward function to use.
        hf_dataset_subset: Optional HF dataset subset/config name (e.g. "main").
        local_dataset_path: If provided, load from this local path instead of
            HuggingFace Hub.
        local_save_dir: Directory to save the output parquet files.
        num_examples: Number of training examples to use. -1 for all. If larger than
            the dataset, all examples are used and a note is printed.
        val_split: Name of a split in the HF dataset to use as validation (e.g. "test").
            If the split doesn't exist, falls back to val_size.
        val_size: Fraction of the train split to hold out for validation (0.0 to 1.0).
            Only used when val_split is None or the requested val_split doesn't exist.
            If 0.0, no validation set is created and None is returned for val_path.

    Returns:
        A tuple of (train_path, val_path, num_train_examples). val_path is None if no
        validation set is created (val_size=0.0 and no valid val_split).

    See: https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
    """
    if local_dataset_path is not None:
        dataset = load_hf_dataset(local_dataset_path, hf_dataset_subset)
    else:
        dataset = load_hf_dataset(hf_dataset_name, hf_dataset_subset)

    train_dataset = dataset["train"]

    # Resolve validation split
    val_dataset = None
    if val_split is not None:
        if val_split in dataset:
            val_dataset = dataset[val_split]
        else:
            print(f"Note: val_split '{val_split}' not found in dataset (available: {list(dataset.keys())}). Falling back to val_size={val_size}.")

    if val_dataset is None and val_size > 0.0:
        split = train_dataset.train_test_split(test_size=val_size, seed=42)
        train_dataset = split["train"]
        val_dataset = split["test"]

    # Limit number of training examples
    if num_examples > 0:
        if num_examples >= len(train_dataset):
            print(f"Note: --num_examples ({num_examples}) >= dataset size ({len(train_dataset)}). Using all {len(train_dataset)} examples.")
        else:
            train_dataset = train_dataset.select(range(num_examples))

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution_gsm8k(answer_raw)
            data = {
                "data_source": hf_dataset_name,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
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

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    os.makedirs(local_save_dir, exist_ok=True)
    train_path = os.path.join(local_save_dir, "train.parquet")
    train_dataset.to_parquet(train_path)

    val_path = None
    if val_dataset is not None:
        val_dataset = val_dataset.map(function=make_map_fn("test"), with_indices=True)
        val_path = os.path.join(local_save_dir, "test.parquet")
        val_dataset.to_parquet(val_path)
        print(f"Dataset saved to {train_path} ({len(train_dataset)} examples) and {val_path} ({len(val_dataset)} examples)")
    else:
        print(f"Dataset saved to {train_path} ({len(train_dataset)} examples). No validation set.")

    return train_path, val_path, len(train_dataset)
