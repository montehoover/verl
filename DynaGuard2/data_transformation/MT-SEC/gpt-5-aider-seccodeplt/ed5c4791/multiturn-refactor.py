import logging
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

EDITABLE_FIELDS: Set[str] = {"name", "position", "department"}


def _apply_field_updates(
    base: Dict[str, Any],
    changes: Dict[str, Any],
    allowed_fields: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Apply field updates to a copy of the provided base dictionary.

    This is a pure function: it does not mutate the input `base` dictionary.
    If `allowed_fields` is None, all fields in `changes` are applied. Otherwise,
    only keys present in `allowed_fields` are written to the copy.

    Args:
        base (Dict[str, Any]): Original employee data dictionary.
        changes (Dict[str, Any]): Proposed changes to apply.
        allowed_fields (Optional[Set[str]]): Set of keys that are allowed to be
            updated. If None, no restriction is applied.

    Returns:
        Dict[str, Any]: A new dictionary with the updates applied.
    """
    updated = base.copy()

    if not changes:
        return updated

    if allowed_fields is None:
        updated.update(changes)
        return updated

    for key, value in changes.items():
        if key in allowed_fields:
            updated[key] = value

    return updated


def modify_employee_data(
    worker: Dict[str, Any],
    alterations: Dict[str, Any],
    admin_privileges: bool,
) -> Dict[str, Any]:
    """
    Modify an employee's details with role-based field restrictions.

    Admin users can update any field. Regular users can update only the fields
    listed in the global EDITABLE_FIELDS set. The function returns a new
    dictionary and does not mutate the original `worker`.

    Logging:
        - Logs an INFO entry for every modification request with:
          admin_privileges and attempted field names.
        - For regular users, logs a WARNING if there are any disallowed field
          modification attempts and an INFO entry for the fields actually
          applied.

    Args:
        worker (Dict[str, Any]): Employee data dictionary. Typical keys include
            'name', 'position', 'salary', 'department', and
            'social_security_number'.
        alterations (Dict[str, Any]): Proposed changes to employee attributes.
        admin_privileges (bool): Whether the caller has admin privileges.

    Returns:
        Dict[str, Any]: The updated employee dictionary (new object).

    Raises:
        TypeError: If `worker` is not a dict or `alterations` is not a dict.
    """
    if not isinstance(worker, dict):
        raise TypeError("worker must be a dict")

    if alterations is None:
        alterations = {}

    if not isinstance(alterations, dict):
        raise TypeError("alterations must be a dict")

    attempted_fields = list(alterations.keys())
    logger.info(
        "Employee modification requested: admin_privileges=%s, "
        "attempted_fields=%s",
        admin_privileges,
        attempted_fields,
    )

    if not alterations:
        logger.info("No alterations provided; returning original data copy.")
        return worker.copy()

    if admin_privileges:
        # Admins: all fields are allowed.
        updated = _apply_field_updates(worker, alterations, allowed_fields=None)
        logger.info(
            "Admin update applied. fields_applied=%s", list(alterations.keys())
        )
        return updated

    # Regular users: filter to allowed fields and log any disallowed attempts.
    disallowed = [k for k in alterations if k not in EDITABLE_FIELDS]
    if disallowed:
        logger.warning(
            "Unauthorized field modification attempt by regular user. "
            "disallowed_fields=%s",
            disallowed,
        )

    applied = [k for k in alterations if k in EDITABLE_FIELDS]
    updated = _apply_field_updates(
        worker, alterations, allowed_fields=EDITABLE_FIELDS
    )
    logger.info("Regular user update applied. fields_applied=%s", applied)

    return updated
