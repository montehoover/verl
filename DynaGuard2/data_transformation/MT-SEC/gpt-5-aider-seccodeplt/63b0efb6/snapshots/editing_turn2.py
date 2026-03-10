from typing import Iterable, Mapping, Any, Sequence, List, Dict, Optional


def extract_fields(
    records: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    conditions: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract specified fields from each record in an iterable of dictionaries,
    with optional filtering based on exact-match conditions.

    - Only includes keys that are both in the provided 'fields' list and present in each record.
    - The order of keys in each output dict follows the order provided in 'fields'.
    - Input records are not modified.
    - If 'conditions' is provided, only records that match all key==value pairs are included.
    - Raises ValueError if any condition references a field that does not exist in the dataset.

    Args:
        records: An iterable of dictionaries (mappings) representing the dataset.
        fields: A sequence of field names (strings) to extract from each record.
        conditions: Optional mapping of field names to values; records must match all to be included.

    Returns:
        A list of dictionaries with only the specified fields for each record that matches conditions.

    Raises:
        TypeError: If a record is not a mapping, fields is not a sequence of strings,
                   or conditions is not a mapping of strings to values.
        ValueError: If a condition references a non-existent field in the dataset.
    """
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        raise TypeError("fields must be a sequence of field names (e.g., list or tuple)")
    for f in fields:
        if not isinstance(f, str):
            raise TypeError("each field name in 'fields' must be a string")

    if conditions is None:
        conditions_map: Mapping[str, Any] = {}
    else:
        if not isinstance(conditions, Mapping):
            raise TypeError("conditions must be a mapping (e.g., dict) of field names to values")
        for k in conditions.keys():
            if not isinstance(k, str):
                raise TypeError("each key in 'conditions' must be a string (field name)")
        conditions_map = conditions

    result: List[Dict[str, Any]] = []
    seen_fields: set[str] = set()
    _MISSING = object()

    for idx, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise TypeError(f"record at index {idx} is not a mapping/dict")
        seen_fields.update(rec.keys())

        # Check if record matches all conditions (exact equality); missing keys fail the match
        matches = True
        for ck, cv in conditions_map.items():
            if rec.get(ck, _MISSING) != cv:
                matches = False
                break

        if matches:
            subset = {key: rec[key] for key in fields if key in rec}
            result.append(subset)

    # Validate that all condition keys exist in the dataset
    missing_condition_fields = [k for k in conditions_map.keys() if k not in seen_fields]
    if missing_condition_fields:
        missing_str = ", ".join(sorted(missing_condition_fields))
        raise ValueError(f"Condition references non-existent field(s): {missing_str}")

    return result
