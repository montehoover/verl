import re

# extract_solution and compute_score are copied verbatim from verl/utils/reward_score/gsm8k.py.
_SOLUTION_CLIP_CHARS = 300

def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score


def gsm8k_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Custom reward function for GSM8K, matching the interface verl expects.

    verl calls custom reward functions with this exact signature. The reward manager
    (verl/workers/reward_manager/naive.py) constructs each argument per-sample from
    the dataset and model rollout:

    Args:
        data_source: The dataset identifier string (e.g. "openai/gsm8k"). Comes from
            the "data_source" field in each row of the training parquet file.
        solution_str: The model's decoded response string for this sample. Decoded from
            the response token IDs using the tokenizer with skip_special_tokens=True.
        ground_truth: The ground truth answer string. Comes from the
            "reward_model.ground_truth" field in each row of the training parquet file.
        extra_info: Optional dict with additional context. Contains "num_turns" (for
            multi-turn rollouts) and "rollout_reward_scores" (reward scores from prior
            turns), plus any "extra_info" field from the dataset row.
        **kwargs: Additional keyword arguments merged from the
            custom_reward_function.reward_kwargs config field.

    Returns:
        A float reward score, or a dict with key "score" (float) plus optional extra
        info keys that get logged.

    The parquet files consumed by this reward function are produced by
    preprocess_dataset_gsm8k() in verl/tomlab/dataset_functions.py, which populates
    the "data_source", "reward_model.ground_truth", and "extra_info" fields.

    See: https://verl.readthedocs.io/en/latest/preparation/reward_function.html
    """
    return compute_score(solution_str, ground_truth)
