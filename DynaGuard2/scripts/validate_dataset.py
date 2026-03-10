import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import Dataset

POLICY_REPLACEMENTS: List[Tuple[str, str]] = [
    ("The product should", "Agent-generated code should"),
    ("The web server should", "Agent-generated web server code should"),
    ("The web application should", "Agent-generated web application code should"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively adjust policy data, optionally collapse the policy field, "
            "and emit a Hugging Face-compatible dataset with a '_validated' suffix."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source JSON/JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional explicit destination path. If omitted, the script "
            "writes to <input_stem>_validated<input_suffix>."
        ),
    )
    return parser.parse_args()


def ensure_metadata_dict(raw_metadata: Any, line_no: int) -> Dict[str, Any]:
    if raw_metadata in ("", None):
        return {}
    assert isinstance(
        raw_metadata, dict
    ), f"Line {line_no}: metadata must be a dict or empty string, got {raw_metadata!r}"
    return dict(raw_metadata)


def load_records_with_line_numbers(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    print(f"Loading records from {path}...")
    records: List[Tuple[int, Dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as src:
        for line_no, raw_line in enumerate(src, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            data = json.loads(raw_line)
            assert isinstance(data, dict), f"Line {line_no}: record must be a dict, got {data!r}"
            records.append((line_no, data))
    return records


def collect_policy_keys(records: Iterable[Tuple[int, Dict[str, Any]]]) -> List[str]:
    keys = set()
    for line_no, record in records:
        policy = record.get("policy")
        if isinstance(policy, dict):
            keys.update(policy.keys())
        else:
            assert isinstance(
                policy, str
            ), f"Line {line_no}: policy must be dict or str, got {type(policy).__name__}"
    return sorted(keys)


def prompt_yes_no(question: str) -> bool:
    while True:
        response = input(f"{question} [y/n]: ").strip().lower()
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print(f"Unrecognized response: {response!r}. Please enter 'y' or 'n'.")


def apply_policy_replacements(
    text: str,
    preferences: Dict[Tuple[str, str], bool],
) -> str:
    updated = text
    for old, new in POLICY_REPLACEMENTS:
        if preferences.get((old, new)):
            updated = updated.replace(old, new)
    return updated


def transform_record(
    record: Dict[str, Any],
    line_no: int,
    move_map: Dict[str, bool],
    collapse_rule: bool,
    replacement_preferences: Dict[Tuple[str, str], bool],
) -> Dict[str, Any]:
    policy = record.get("policy")

    if isinstance(policy, dict):
        metadata_dict = ensure_metadata_dict(record.get("metadata"), line_no)

        for key, should_move in move_map.items():
            if not should_move or key not in policy:
                continue
            assert (
                key not in metadata_dict
            ), f"Line {line_no}: metadata already contains {key!r}, cannot override"
            metadata_dict[key] = policy.pop(key)

        record["metadata"] = metadata_dict

        if "rule" in policy:
            rule_value = policy["rule"]
            assert isinstance(
                rule_value, str
            ), f"Line {line_no}: rule must be a string, got {rule_value!r}"
            policy["rule"] = apply_policy_replacements(rule_value, replacement_preferences)

        if collapse_rule and "rule" in policy:
            rule_value = policy.pop("rule")
            assert isinstance(
                rule_value, str
            ), f"Line {line_no}: rule must be a string before collapsing, got {rule_value!r}"
            assert (
                not policy
            ), f"Line {line_no}: cannot collapse policy because fields remain: {list(policy.keys())}"
            record["policy"] = rule_value
            return record

        record["policy"] = policy
        return record

    assert isinstance(
        policy, str
    ), f"Line {line_no}: policy must be dict or str, got {type(policy).__name__}"
    record["policy"] = apply_policy_replacements(policy, replacement_preferences)
    return record


def transform_records(
    records: List[Tuple[int, Dict[str, Any]]],
    move_map: Dict[str, bool],
    collapse_rule: bool,
    replacement_preferences: Dict[Tuple[str, str], bool],
) -> List[Dict[str, Any]]:
    transformed: List[Dict[str, Any]] = []
    for line_no, record in records:
        updated_record = transform_record(
            record,
            line_no,
            move_map,
            collapse_rule,
            replacement_preferences,
        )
        transformed.append(updated_record)
    return transformed


def iter_policy_texts(record: Dict[str, Any], line_no: int) -> Iterable[str]:
    policy = record.get("policy")
    if isinstance(policy, str):
        yield policy
        return
    if isinstance(policy, dict):
        rule_value = policy.get("rule")
        if isinstance(rule_value, str):
            yield rule_value
        return
    raise AssertionError(
        f"Line {line_no}: policy must be dict or str, got {type(policy).__name__}"
    )


def count_replacement_occurrences(
    records: List[Tuple[int, Dict[str, Any]]],
) -> Dict[Tuple[str, str], int]:
    counts = {pair: 0 for pair in POLICY_REPLACEMENTS}
    for line_no, record in records:
        for text in iter_policy_texts(record, line_no):
            for pair in POLICY_REPLACEMENTS:
                counts[pair] += text.count(pair[0])
    return counts


def validate_and_save_dataset(
    records: List[Dict[str, Any]],
    destination: Path,
) -> None:
    print(f"Building Hugging Face Dataset from {len(records)} records...")
    dataset = Dataset.from_list(records)
    print("Hugging Face Dataset built successfully.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving validated dataset to {destination}...")
    dataset.to_json(
        destination.as_posix(),
        orient="records",
        lines=False,
        indent=2,
        force_ascii=False,
    )
    print(f"Saved {len(records)} transformed records to {destination}.")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.with_name(f"{input_path.stem}_validated.json")

    assert input_path.exists(), f"Input path does not exist: {input_path}"
    assert input_path.is_file(), f"Input path is not a file: {input_path}"

    records = load_records_with_line_numbers(input_path)
    policy_keys = collect_policy_keys(records)
    print(f"Discovered policy keys: {', '.join(policy_keys) if policy_keys else '(none)'}")
    replacement_counts = count_replacement_occurrences(records)

    move_map: Dict[str, bool] = {}
    for key in policy_keys:
        if key == "rule":
            continue
        move_map[key] = prompt_yes_no(
            f"Move policy field '{key}' into metadata"
        )

    print("Configuring policy text replacements...")
    replacement_preferences: Dict[Tuple[str, str], bool] = {}
    for old, new in POLICY_REPLACEMENTS:
        count = replacement_counts[(old, new)]
        if count == 0:
            replacement_preferences[(old, new)] = False
            print(
                f"Skipping replacement '{old}' -> '{new}': no occurrences found in policy text."
            )
            continue
        replacement_preferences[(old, new)] = prompt_yes_no(
            f"Replace '{old}' with '{new}' in policy text (would modify {count} occurrence(s))"
        )

    collapse_rule = False
    if "rule" in policy_keys:
        collapse_rule = prompt_yes_no(
            "Collapse policy to the rule string (removes 'rule' key)"
        )

    transformed_records = transform_records(
        records,
        move_map,
        collapse_rule,
        replacement_preferences,
    )
    print(f"Transformed {len(transformed_records)} records from {input_path}.")
    validate_and_save_dataset(transformed_records, output_path)


if __name__ == "__main__":
    main()

