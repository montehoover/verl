import os
import re
import datasets


def extract_solution_gsm8k(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def preprocess_dataset_gsm8k(hf_dataset_name="openai/gsm8k",local_dataset_path=None, local_save_dir="data/gsm8k"):
    """
    Preprocess the GSM8k dataset to parquet format.
    """
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(hf_dataset_name, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

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
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    
    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print(f"Dataset downloaded and saved to {train_path} and {test_path}")
    return train_path, test_path


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
