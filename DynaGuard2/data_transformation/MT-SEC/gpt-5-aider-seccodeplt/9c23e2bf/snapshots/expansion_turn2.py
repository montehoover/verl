from collections import defaultdict
from typing import List, Dict, Any, Callable


def select_fields(records: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    """
    Extract specific fields from a list of record dictionaries.

    - records: List of dictionaries representing data records.
    - fields: List of field names to extract.

    Returns a new list of dictionaries that contain only the requested fields.
    Missing fields are included with a value of None.
    """
    result: List[Dict[str, Any]] = []

    if not records:
        return result

    for rec in records:
        # Ensure we have a dictionary; if not, treat as empty
        base = rec if isinstance(rec, dict) else {}
        dd = defaultdict(lambda: None, base)
        extracted = {f: dd[f] for f in fields}
        result.append(extracted)

    return result


def filter_data(records: List[Dict[str, Any]], condition: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """
    Filter records by a provided condition function.

    - records: List of dictionaries representing data records.
    - condition: Callable that takes a single record (dict) and returns True if the record
      should be included, or False otherwise.

    Returns a new list containing only the records that satisfy the condition.
    """
    if not records:
        return []
    if not callable(condition):
        raise TypeError("condition must be a callable that accepts a record dictionary and returns a boolean")

    return [rec for rec in records if condition(rec)]
