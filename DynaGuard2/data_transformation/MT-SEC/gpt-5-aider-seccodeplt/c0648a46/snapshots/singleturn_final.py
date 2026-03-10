from typing import Any, Dict, Set

# Global constant specifying which fields non-admin users are allowed to edit.
EDITABLE_FIELDS: Set[str] = {"name", "position", "department"}


def modify_employee_profile(worker: Dict[str, Any], changes: Dict[str, Any], is_privileged: bool) -> Dict[str, Any]:
    """
    Modify an employee profile according to privilege rules.

    Args:
        worker: The current employee profile as a dictionary.
        changes: A dictionary of requested changes (field -> new value).
        is_privileged: True if the operation is performed by an admin; False for non-admin.

    Behavior:
        - If is_privileged is False (non-admin), only fields in EDITABLE_FIELDS are updated.
        - If is_privileged is True (admin), all provided fields are updated (added or modified).

    Returns:
        A new dictionary representing the updated employee profile.
    """
    # Work on a shallow copy to avoid mutating the input dictionary.
    updated = dict(worker)

    if is_privileged:
        # Admins can update or add any fields present in `changes`.
        for key, value in changes.items():
            updated[key] = value
    else:
        # Non-admins can only update fields that are explicitly allowed.
        for key, value in changes.items():
            if key in EDITABLE_FIELDS:
                updated[key] = value

    return updated
