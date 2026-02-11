import os
import re
import datasets

GSM8K_DATASET_NAME = "openai/gsm8k"


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


# ---------------------------------------------------------------------------
# Generic entrypoint — contains all shared boilerplate
# ---------------------------------------------------------------------------

def list_dataset_map_functions():
    excluded_names = {"preprocess_dataset"}
    names = []
    for name, value in globals().items():
        if name.startswith("preprocess_") and name not in excluded_names and callable(value):
            names.append(name)
    return sorted(names)


def get_dataset_map_function(map_fn_name):
    assert isinstance(map_fn_name, str), f"Expected map function name string, got {type(map_fn_name)}: {map_fn_name!r}"
    map_fn = globals().get(map_fn_name)
    assert callable(map_fn), (
        f"Unknown map function '{map_fn_name}'. Available: {', '.join(list_dataset_map_functions())}"
    )
    return map_fn


def preprocess_dataset(
    map_fn_name,
    hf_dataset_name,
    hf_dataset_subset=None,
    local_dataset_path=None,
    local_save_dir="data",
    num_examples=-1,
    val_split=None,
    val_size=0.0,
):
    """Download, preprocess, and save a dataset into verl's expected parquet format.

    This is the generic entrypoint that handles all shared boilerplate: loading
    the dataset, resolving validation splits, limiting examples, applying the
    per-row transform, and saving parquet files.

    Args:
        map_fn_name: Name of the map function in this file that returns a
            ``split -> process_fn`` callable. Examples: ``preprocess_gsm8k`` or
            ``preprocess_dynabench``.
        hf_dataset_name: HuggingFace dataset identifier.
        hf_dataset_subset: Optional HF dataset subset/config name.
        local_dataset_path: If provided, load from this local path instead of
            HuggingFace Hub.
        local_save_dir: Directory to save the output parquet files.
        num_examples: Number of training examples to use. -1 for all.
        val_split: Name of a split in the HF dataset to use as validation.
        val_size: Fraction of the train split to hold out for validation.

    Returns:
        A tuple of (train_path, val_path, num_train_examples). val_path is None
        if no validation set is created.
    """
    map_fn_builder = get_dataset_map_function(map_fn_name)
    make_map_fn = map_fn_builder()

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


# ---------------------------------------------------------------------------
# Per-dataset map functions — just the row transform logic
# ---------------------------------------------------------------------------

def preprocess_gsm8k():
    """Return a ``make_map_fn`` closure for GSM8K rows."""
    dataset_name = GSM8K_DATASET_NAME
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw + " " + instruction_following
            answer_raw = example.pop("answer")
            solution = extract_solution_gsm8k(answer_raw)
            data = {
                "data_source": dataset_name,
                "prompt": [{"role": "user", "content": question}],
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
    return make_map_fn


