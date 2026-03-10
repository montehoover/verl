from collections import defaultdict
from typing import Any, Dict, Iterable, List
from typing import Callable, List, Dict

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


def apply_filter(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Return only records that satisfy the given condition.
    The condition can be any callable that accepts a record and returns a bool.
    Logical operations can be composed in the callable, e.g.:
        lambda r: cond1(r) and (cond2(r) or not cond3(r))
    """
    if not records:
        return []

    return [record for record in records if condition(record)]
