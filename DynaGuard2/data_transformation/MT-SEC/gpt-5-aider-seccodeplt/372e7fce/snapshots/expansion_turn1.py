from collections import defaultdict
from typing import Iterable, Mapping, Sequence, Dict, Any, List

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
