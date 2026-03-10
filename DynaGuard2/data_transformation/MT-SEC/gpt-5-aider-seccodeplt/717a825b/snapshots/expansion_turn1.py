from collections import defaultdict
from typing import Any, Dict, List


def select_fields(records: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    """
    Extract a subset of fields from each record in a list of dictionaries.

    - records: list of dictionaries representing records
    - fields: list of field names to select from each record

    Returns a list of dictionaries containing only the specified fields.
    If a field is missing in a record, its value will be None.
    """
    if records is None:
        return []

    result: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            record = {}
        selected = defaultdict(lambda: None)
        for field in fields:
            selected[field] = record.get(field)
        result.append(dict(selected))

    return result
