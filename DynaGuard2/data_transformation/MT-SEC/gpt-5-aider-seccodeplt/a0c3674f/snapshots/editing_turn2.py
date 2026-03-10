from typing import Any, Dict, Set

# Predefined set of fields that can be updated without special permission.
# Populate this set with the keys you want to allow.
ALLOWED_FIELDS: Set[Any] = set()

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
