from typing import Any, Dict, List, Tuple


def modify_employee_profile(worker: Dict[str, Any], changes: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Update an employee profile with given changes and produce a log of each field altered.

    Args:
        worker: The current employee profile as a dictionary.
        changes: The fields to update as a dictionary.

    Returns:
        A tuple of:
          - updated employee profile (dict)
          - change log (list of dict), where each entry includes:
                {
                    "field": <field name>,
                    "old": <previous value or None if not present>,
                    "new": <new value>
                }
    """
    if not isinstance(worker, dict):
        raise TypeError("worker must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    updated = dict(worker)  # shallow copy to avoid mutating the original
    log: List[Dict[str, Any]] = []

    for field, new_value in changes.items():
        if field in updated:
            old_value = updated[field]
            if old_value != new_value:
                updated[field] = new_value
                log.append({"field": field, "old": old_value, "new": new_value})
        else:
            updated[field] = new_value
            log.append({"field": field, "old": None, "new": new_value})

    return updated, log
