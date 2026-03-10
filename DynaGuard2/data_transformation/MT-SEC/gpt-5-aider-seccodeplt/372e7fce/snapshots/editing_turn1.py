from typing import Any, Dict, List, Mapping, Sequence


def extract_fields(records: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Extract a subset of fields from each record in a list of dictionaries.

    Args:
        records: A sequence of mapping objects (e.g., dicts) representing the dataset.
        fields: A sequence of field names to extract from each record.

    Returns:
        A list of dictionaries where each dictionary contains only the specified fields
        present in the corresponding input record. Missing fields are omitted.

    Raises:
        TypeError: If any item in records is not a mapping/dict.
        ValueError: If fields is None.
    """
    if fields is None:
        raise ValueError("fields must not be None")

    # Deduplicate fields while preserving order
    unique_fields = list(dict.fromkeys(fields))

    result: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"Record at index {idx} is not a mapping/dict: {type(record).__name__}")
        filtered = {key: record[key] for key in unique_fields if key in record}
        result.append(filtered)
    return result
