from collections import defaultdict
from typing import Any, Dict, Iterable, List

def select_columns(records: List[Dict[str, Any]], columns: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Extract only the specified columns from each record.
    Missing columns are included with a value of None.
    """
    if not records:
        return []

    cols = list(columns)
    result: List[Dict[str, Any]] = []

    for record in records:
        safe_record = defaultdict(lambda: None, record or {})
        selected = {col: safe_record[col] for col in cols}
        result.append(selected)

    return result
