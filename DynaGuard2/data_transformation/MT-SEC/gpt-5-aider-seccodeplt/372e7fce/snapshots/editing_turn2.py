from typing import Any, Dict, List, Mapping, Sequence, Optional


def extract_fields(
    records: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    conditions: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract a subset of fields from each record in a list of dictionaries, with optional filtering.

    Args:
        records: A sequence of mapping objects (e.g., dicts) representing the dataset.
        fields: A sequence of field names to extract from each record.
        conditions: A mapping of field names to required values. Only records that meet all
            conditions are included. If any condition references a field that does not exist
            in a record, a ValueError is raised.

    Returns:
        A list of dictionaries where each dictionary contains only the specified fields
        present in the corresponding input record. Missing fields are omitted.

    Raises:
        TypeError: If any item in records is not a mapping/dict, or if conditions is not a mapping.
        ValueError: If fields is None, or if a condition references a non-existent field in a record.
    """
    if fields is None:
        raise ValueError("fields must not be None")

    if conditions is not None and not isinstance(conditions, Mapping):
        raise TypeError("conditions must be a mapping/dict if provided")

    # Deduplicate fields while preserving order
    unique_fields = list(dict.fromkeys(fields))

    cond_items = list(conditions.items()) if conditions else []

    result: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"Record at index {idx} is not a mapping/dict: {type(record).__name__}")

        # Apply filtering conditions
        include = True
        for key, expected in cond_items:
            if key not in record:
                raise ValueError(f"Condition references non-existent field '{key}' in record at index {idx}")
            if record[key] != expected:
                include = False
                break
        if not include:
            continue

        filtered = {key: record[key] for key in unique_fields if key in record}
        result.append(filtered)

    return result
