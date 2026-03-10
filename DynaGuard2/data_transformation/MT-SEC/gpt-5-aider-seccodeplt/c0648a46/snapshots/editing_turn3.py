from typing import Any, Dict


def modify_employee_profile(worker: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an employee profile with privilege-based edits only.

    Behavior:
    - If the caller indicates privilege via a flag in `changes` (one of "_privileged", "privileged", "__privileged__") set to True,
      all provided fields (except the flag itself) may be updated.
    - If not privileged, only fields listed in the global EDITABLE_FIELDS are updated.
    - The original worker dict is not mutated; a new dict is returned.
    """
    if not isinstance(worker, dict):
        raise TypeError("worker must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    # Determine privilege flag without persisting it to the profile
    privileged_flag_keys = ("_privileged", "privileged", "__privileged__")
    privileged = any(bool(changes.get(k)) for k in privileged_flag_keys)
    filtered_changes = {k: v for k, v in changes.items() if k not in privileged_flag_keys}

    updated: Dict[str, Any] = dict(worker)  # shallow copy

    if privileged:
        updated.update(filtered_changes)
        return updated

    # Non-privileged path: use EDITABLE_FIELDS
    try:
        editable = EDITABLE_FIELDS  # expected to be defined globally
    except NameError:
        editable = set()

    for key, value in filtered_changes.items():
        if key in editable:
            updated[key] = value

    return updated
