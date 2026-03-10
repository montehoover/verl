from typing import List, Dict, Any, Optional, Callable, Mapping, Union


FilterType = Optional[Union[Callable[[Dict[str, Any]], bool], Mapping[str, Any]]]


def _matches_filters(record: Dict[str, Any], filters: FilterType) -> bool:
    if filters is None:
        return True
    if callable(filters):
        return bool(filters(record))
    if isinstance(filters, Mapping):
        for key, value in filters.items():
            if key not in record or record[key] != value:
                return False
        return True
    raise ValueError("filters must be None, a callable, or a mapping of field names to values")


def extract_fields(
    records: List[Dict[str, Any]],
    fields: List[str],
    filters: FilterType = None
) -> List[Dict[str, Any]]:
    """
    Extract specified fields from a list of record dictionaries, with optional filtering.

    Args:
        records: A list of dictionaries representing records.
        fields: A list of field names to extract from each record.
        filters: Optional filter conditions. Can be:
                 - None: no filtering
                 - Callable[[Dict[str, Any]], bool]: a predicate that returns True if the record should be included
                 - Mapping[str, Any]: equality-based filters; a record is included only if record[k] == v for all items

    Returns:
        A new list of dictionaries containing only the specified fields from records
        that satisfy the filter conditions.

    Raises:
        ValueError: If any item in records is not a dict, if a requested field is not
                    present in a matching record, or if filters is of an unsupported type.
    """
    result: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {idx} is not a dictionary")

        if not _matches_filters(record, filters):
            continue

        missing = [field for field in fields if field not in record]
        if missing:
            raise ValueError(
                f"Record at index {idx} is missing required field(s): {', '.join(missing)}"
            )

        result.append({field: record[field] for field in fields})

    return result
