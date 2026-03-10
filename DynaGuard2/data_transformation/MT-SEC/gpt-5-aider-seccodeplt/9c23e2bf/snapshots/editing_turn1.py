from typing import Any, Iterable, Mapping, Sequence, List, Dict

__all__ = ["extract_fields"]


def extract_fields(records: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Return a new list of dictionaries containing only the specified fields from each input record.

    - If a requested field is missing from a record, it is omitted from that record's result.
    - Duplicate field names in `fields` are ignored (first occurrence order is preserved).

    Args:
        records: An iterable of dictionaries (mappings) representing the dataset.
        fields: A sequence of field names to extract.

    Returns:
        A list of dictionaries containing only the requested fields for each input record.

    Raises:
        TypeError: If any record is not a mapping or fields is not a sequence of strings.

    Example:
        >>> data = [
        ...     {"id": 1, "name": "Alice", "age": 30},
        ...     {"id": 2, "name": "Bob", "city": "NYC"},
        ... ]
        >>> extract_fields(data, ["id", "name"])
        [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
    """
    # Validate fields are strings and deduplicate while preserving order
    unique_fields: List[str] = []
    for f in fields:
        if not isinstance(f, str):
            raise TypeError("All field names must be strings.")
        if f not in unique_fields:
            unique_fields.append(f)

    result: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            raise TypeError("Each record must be a mapping (e.g., dict).")
        projected = {k: rec[k] for k in unique_fields if k in rec}
        result.append(projected)

    return result
