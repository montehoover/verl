from typing import Any, Dict, Set

# Predefined set of fields that can be updated without special permission.
# Populate this set with the keys you want to allow.
ALLOWED_FIELDS: Set[Any] = set()

# Fields that non-admin users are allowed to edit in employee records.
EDITABLE_FIELDS: Set[str] = {'name', 'position', 'department'}

def update_record(data: Dict[Any, Any], modifications: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Return a new dictionary combining data with modifications.

    - Shallow merge: keys in modifications overwrite keys in data.
    - Does not mutate the original 'data' dictionary.
    """
    updated = dict(data)
    updated.update(modifications)
    return updated

def restricted_update(
    data: Dict[Any, Any],
    modifications: Dict[Any, Any],
    has_permission: bool
) -> Dict[Any, Any]:
    """
    Return a new dictionary combining data with modifications, with permission control.

    - If has_permission is True: apply all modifications (shallow merge).
    - If has_permission is False: only apply modifications for keys in ALLOWED_FIELDS.
    - Does not mutate the original 'data' dictionary.
    """
    updated = dict(data)
    if has_permission:
        updated.update(modifications)
    else:
        for key, value in modifications.items():
            if key in ALLOWED_FIELDS:
                updated[key] = value
    return updated

def modify_staff_info(
    staff: Dict[str, Any],
    changes: Dict[str, Any],
    admin_status: bool
) -> Dict[str, Any]:
    """
    Return an updated copy of the staff dictionary with permission-aware changes.

    - If admin_status is True: apply all changes (shallow merge).
    - If admin_status is False: only apply changes for keys in EDITABLE_FIELDS.
    - Does not mutate the original 'staff' dictionary.
    """
    updated = dict(staff)
    if admin_status:
        updated.update(changes)
    else:
        for key, value in changes.items():
            if key in EDITABLE_FIELDS:
                updated[key] = value
    return updated
