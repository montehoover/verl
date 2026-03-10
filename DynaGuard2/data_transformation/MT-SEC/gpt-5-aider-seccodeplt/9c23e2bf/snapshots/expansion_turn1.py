from collections import defaultdict
from typing import List, Dict, Any


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
