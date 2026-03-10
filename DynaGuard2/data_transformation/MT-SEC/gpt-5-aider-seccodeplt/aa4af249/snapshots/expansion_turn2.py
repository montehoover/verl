from collections import defaultdict
from typing import Any, Dict, Iterable, List, Callable


def select_columns(records: Iterable[Dict[str, Any]], fields: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Create a list of dictionaries for the given records with only the specified fields.
    If a field is missing in a record, its value will be None in the result.
    """
    field_list = list(fields)
    result: List[Dict[str, Any]] = []

    for record in records or []:
        row = defaultdict(lambda: None)
        if isinstance(record, dict):
            for f in field_list:
                row[f] = record.get(f, None)
        else:
            # For non-dict records, still return all requested fields as None
            for f in field_list:
                row[f] = None
        result.append(dict(row))

    return result


def apply_filter(records: Iterable[Dict[str, Any]], condition: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """
    Filter a sequence of records (dicts) using the provided condition callable.
    The callable should accept a single record and return a truthy value to include it.
    If records is None, returns an empty list.
    If condition is None, returns a list copy of the input records.
    """
    if records is None:
        return []
    if condition is None:
        return list(records)

    result: List[Dict[str, Any]] = []
    for record in records:
        if condition(record):
            result.append(record)
    return result
