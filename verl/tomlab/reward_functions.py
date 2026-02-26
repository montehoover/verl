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


def extract_xml_answer(text, opening_tag="<answer>", closing_tag="</answer>"):
    """Extract the text between XML-style opening and closing tags.

    Splits on the opening tag and takes the last segment, then splits on the
    closing tag and takes the first segment. This handles cases where tags
    appear multiple times by taking the last opening tag match.

    Args:
        text: The full text to extract from.
        opening_tag: The opening XML tag (e.g. "<answer>").
        closing_tag: The closing XML tag (e.g. "</answer>").

    Returns:
        The stripped text between the tags, or the original text stripped if
        tags are not found.
    """
    answer = text.split(opening_tag.strip())[-1]
    answer = answer.split(closing_tag.strip())[0]
    return answer.strip()


def dynabench_correctness_reward(solution_str, ground_truth, label_opening, label_closing, points=0.5):
    """Score based on whether the predicted label matches the ground truth label."""
    prediction = extract_xml_answer(solution_str, label_opening, label_closing)
    return points if prediction == ground_truth else 0.0


def dynabench_label_format_reward(solution_str, label_opening, label_closing, pos_label, neg_label, points=0.1):
    """Score based on whether the extracted label is exactly PASS or FAIL."""
    prediction = extract_xml_answer(solution_str, label_opening, label_closing)
    return points if prediction in [pos_label, neg_label] else 0.0


def dynabench_strict_format_reward(solution_str, label_opening, label_closing, points=0.1):
    """Score based on strict XML format with newlines before/after tags."""
    pattern = fr"^\s*{re.escape(label_opening)}\n.*?\n{re.escape(label_closing)}\s*"
    match = re.search(pattern, solution_str, re.DOTALL)
    return points if match else 0.0


def dynabench_soft_format_reward(solution_str, label_opening, label_closing, points=0.1):
    """Score based on soft XML format (flexible whitespace around tags)."""
    pattern = fr"{re.escape(label_opening)}\s*.*?\s*{re.escape(label_closing)}"
    match = re.search(pattern, solution_str, re.DOTALL)
    return points if match else 0.0


def dynabench_xml_count_reward(solution_str, label_opening, label_closing, rules_opening, rules_closing, points=0.05):
    """Score based on each expected XML tag appearing exactly once."""
    reward = 0.0
    if solution_str.count(label_opening) == 1:
        reward += points
    if solution_str.count(label_closing) == 1:
        reward += points
    if solution_str.count(rules_opening) == 1:
        reward += points
    if solution_str.count(rules_closing) == 1:
        reward += points
    return reward


def dynabench_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Custom reward function for DynaBench, matching the interface verl expects.

    Scores the model's response against the ground truth label (PASS/FAIL) using
    a combination of correctness and format rewards. The reward breakdown is:

        - Correctness (label matches ground truth): 0.50
        - Label format (extracted label is exactly PASS or FAIL): 0.10
        - Strict format (XML tags with proper newlines): 0.10
        - Soft format (XML tags with flexible whitespace): 0.10
        - XML tag count (each tag appears exactly once): 0.05 per tag (up to 0.20)

    verl calls custom reward functions with this exact signature. The reward manager
    (verl/workers/reward_manager/naive.py) constructs each argument per-sample from
    the dataset and model rollout:

    Args:
        data_source: The dataset identifier string (e.g. "tomg-group-umd/dynabench").
            Comes from the "data_source" field in each row of the training parquet file.
        solution_str: The model's decoded response string for this sample. Decoded from
            the response token IDs using the tokenizer with skip_special_tokens=True.
        ground_truth: The ground truth label string ("PASS" or "FAIL"). Comes from the
            "reward_model.ground_truth" field in each row of the training parquet file.
        extra_info: Optional dict with additional context. Contains "num_turns" (for
            multi-turn rollouts) and "rollout_reward_scores" (reward scores from prior
            turns), plus any "extra_info" field from the dataset row.
        **kwargs: Additional keyword arguments merged from the
            custom_reward_function.reward_kwargs config field.

    Returns:
        A float reward score between 0.0 and 1.0.

    The parquet files consumed by this reward function are produced by
    preprocess_dataset_dynabench() in verl/tomlab/dataset_functions.py, which populates
    the "data_source", "reward_model.ground_truth", and "extra_info" fields.

    See: https://verl.readthedocs.io/en/latest/preparation/reward_function.html
    """
    label_opening = "<answer>"
    label_closing = "</answer>"
    rules_opening = "<rules_violated>"
    rules_closing = "</rules_violated>"
    pos_label = "FAIL"
    neg_label = "PASS"

    print(f"solution_str: {solution_str} THEEND")
    print(f"ground_truth: {ground_truth} THEEND")

    correctness = dynabench_correctness_reward(solution_str, ground_truth, label_opening, label_closing, points=0.50)
    label_fmt = dynabench_label_format_reward(solution_str, label_opening, label_closing, pos_label, neg_label, points=0.10)
    strict_fmt = dynabench_strict_format_reward(solution_str, label_opening, label_closing, points=0.10)
    soft_fmt = dynabench_soft_format_reward(solution_str, label_opening, label_closing, points=0.10)
    xml_count = dynabench_xml_count_reward(solution_str, label_opening, label_closing, rules_opening, rules_closing, points=0.05)

    return correctness + label_fmt + strict_fmt + soft_fmt + xml_count