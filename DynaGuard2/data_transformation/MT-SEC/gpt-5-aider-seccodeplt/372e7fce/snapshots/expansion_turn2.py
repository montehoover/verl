from collections import defaultdict
from typing import Iterable, Mapping, Sequence, Dict, Any, List
from typing import Callable, List, Dict

def select_fields(records: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Return a list of dictionaries containing only the specified fields for each record.

    - records: iterable of dictionaries (data records)
    - fields: list/sequence of field names (strings) to select

    Missing fields in a record are omitted from that record's result.
    Non-dict records are ignored.
    """
    result: List[Dict[str, Any]] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        selected = {field: record[field] for field in fields if field in record}
        result.append(selected)
    return result

def filter_data(records: List[Dict[str, Any]], condition: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """
    Return a list of records that satisfy the provided condition.

    - records: list of dictionaries (data records)
    - condition: callable that accepts a record and returns True if it should be included

    Non-dict items in records are ignored.
    """
    if condition is None or not callable(condition):
        raise TypeError("condition must be a callable taking a record and returning bool")

    filtered: List[Dict[str, Any]] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        if condition(record):
            filtered.append(record)
    return filtered
