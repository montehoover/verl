from typing import Iterable, Mapping, Any, List, Dict, Sequence


def extract_fields(records: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Create a new list of dictionaries containing only the specified fields from each record.

    Behavior:
    - Only includes keys that exist in each record (missing keys are skipped).
    - The order of keys in the output dictionaries follows the order provided in `fields`.
    - Duplicate field names in `fields` are de-duplicated, preserving their first occurrence.

    Args:
        records: An iterable of mapping objects (e.g., list of dicts).
        fields: A sequence of field names to include.

    Returns:
        A list of dictionaries, each containing only the specified fields present in the original records.

    Raises:
        TypeError: If any record is not a mapping/dict.
    """
    if records is None:
        return []
    if fields is None:
        return []

    # De-duplicate fields while preserving order
    seen = set()
    ordered_fields: List[str] = []
    for f in fields:
        if f not in seen:
            seen.add(f)
            ordered_fields.append(f)

    result: List[Dict[str, Any]] = []
    for rec in records:
        if rec is None:
            continue
        if not isinstance(rec, Mapping):
            raise TypeError(f"Each record must be a mapping/dict, got {type(rec).__name__}")
        filtered = {k: rec[k] for k in ordered_fields if k in rec}
        result.append(filtered)

    return result
