from typing import Iterable, Mapping, Any, Sequence, List, Dict


def extract_fields(records: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Extract specified fields from each record in a list (or iterable) of dictionaries.

    - Only includes keys that are both in the provided 'fields' list and present in each record.
    - The order of keys in each output dict follows the order provided in 'fields'.
    - Input records are not modified.

    Args:
        records: An iterable of dictionaries (mappings) representing the dataset.
        fields: A sequence of field names (strings) to extract from each record.

    Returns:
        A list of dictionaries with only the specified fields for each record.

    Raises:
        TypeError: If a record is not a mapping or fields is not a sequence of strings.
    """
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        raise TypeError("fields must be a sequence of field names (e.g., list or tuple)")

    # Optional: ensure each field name is a string
    for f in fields:
        if not isinstance(f, str):
            raise TypeError("each field name in 'fields' must be a string")

    result: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise TypeError(f"record at index {idx} is not a mapping/dict")
        # Build dict in the order specified by fields; only include existing keys
        subset = {key: rec[key] for key in fields if key in rec}
        result.append(subset)
    return result
