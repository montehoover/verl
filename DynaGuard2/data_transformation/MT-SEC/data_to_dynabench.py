#!/usr/bin/env python3
"""
Convert a filtered subset of MT-SEC transformed data into Dynabench format.

Inputs:
- original data (transformed): /space3/yangfc/benchmarks/MT-SEC/transformed-secode-allsamples.json
- filtered selection: /space3/yangfc/benchmarks/MT-SEC/filtered_selection.json

Output:
- Dynabench JSONL: /space3/yangfc/benchmarks/DynaGuard/dynabench/test_dynabench.json

By default, the transcript comes from the singleturn content. You may choose to
emit the selected multiturn transcript instead via --content multiturn.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert MT-SEC selection to Dynabench format")
    p.add_argument(
        "--original",
        default="/space3/yangfc/benchmarks/MT-SEC/transformed-secode-allsamples.json",
        help="Path to transformed original data JSON",
    )
    p.add_argument(
        "--selection",
        default="/space3/yangfc/benchmarks/MT-SEC/filtered_selection.json",
        help="Path to filtered selection JSON (pass/fail groups)",
    )
    p.add_argument(
        "--output",
        default="/space3/yangfc/benchmarks/MT-SEC/dyna-format/singleturn_mtsec_dynabench.json",
        help="Output Dynabench JSONL path",
    )
    p.add_argument(
        "--content",
        choices=["singleturn", "multiturn"],
        default="singleturn",
        help="Which transcript to emit for each sample",
    )
    return p.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_label(capability: str, safety: str) -> str:
    # Caller ensures capability == "1" via selection, but keep logic explicit
    if capability == "1" and safety == "1":
        return "PASS"
    if capability == "1" and safety == "0":
        return "FAIL"
    # Fallback (should not occur with the provided selection)
    return "FAIL"


def build_records(
    original: Dict[str, Any],
    selection: Dict[str, List[Dict[str, Any]]],
    content_mode: str = "singleturn",
) -> List[Dict[str, Any]]:
    records = []

    # Flatten pass/fail groups but keep label from original results
    items = []
    for group in ("pass", "fail"):
        for entry in selection.get(group, []):
            items.append(entry)

    for entry in items:
        sid = entry["id"]
        mt_type = entry["multiturn"]
        sample = original.get(sid)
        if not sample:
            continue

        # Policy from original
        policy = sample.get("policy", {})
        cwe_id = str(policy.get("CWE_ID", ""))
        rule = policy.get("rule", "")

        # Transcript selection
        if content_mode == "singleturn":
            transcript = sample.get("singleturn", "")
        else:
            transcript = sample.get("multiturn", {}).get(mt_type, "")

        # Label mapping from singleturn results (equal to chosen multiturn by construction)
        st_res = sample.get("results", {}).get("singleturn", {})
        label = map_label(str(st_res.get("capability", "")), str(st_res.get("safety", "")))

        rec = {
            "policy": {
                "CWE_ID": cwe_id,
                "rule": rule,
            },
            "transcript": transcript,
            "label": label,
            "metadata": "",
        }
        records.append(rec)

    return records


def write_jsonl(records: List[Dict[str, Any]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    original = load_json(args.original)
    selection = load_json(args.selection)
    records = build_records(original, selection, content_mode=args.content)
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

