from collections import defaultdict
from typing import List, Dict, Any, Iterable, Optional


def select_columns(records: Optional[Iterable[Dict[str, Any]]], fields: Optional[Iterable[str]]) -> List[Dict[str, Any]]:
    """
    Return a list of dictionaries each containing only the specified fields from the input records.

    - records: iterable of dictionaries (each representing a record)
    - fields: iterable of strings representing field names to select

    Behavior:
    - If a field is not present in a record, it is simply omitted (no error raised).
    - If records is None, returns an empty list.
    - If fields is None or empty, returns a list of empty dictionaries (one per record).
    - Non-dictionary items in records are treated as empty dictionaries.
    - The order of keys in each resulting dictionary follows the order of the provided fields.
    """
    if records is None:
        return []

    fields = list(fields or [])
    result: List[Dict[str, Any]] = []

    for rec in records:
        rec_dict = rec if isinstance(rec, dict) else {}
        selected: Dict[str, Any] = {}

        for f in fields:
            if f in rec_dict:
                selected[f] = rec_dict[f]

        result.append(selected)

    return result
