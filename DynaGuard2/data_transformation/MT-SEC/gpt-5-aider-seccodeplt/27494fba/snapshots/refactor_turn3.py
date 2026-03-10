import logging

EDITABLE_FIELDS = {"name", "position", "department"}

logger = logging.getLogger(__name__)


def adjust_employee_profile(emp: dict, changes: dict, has_admin_rights: bool) -> dict:
    """
    Modify an employee's details.

    - Admin users (has_admin_rights=True) can update any provided fields.
    - Regular users can only update fields included in EDITABLE_FIELDS.

    Args:
        emp: Existing employee dictionary.
        changes: Proposed modifications to the employee.
        has_admin_rights: Whether the updater has administrative rights.

    Returns:
        A new dictionary representing the updated employee.
    """
    if not isinstance(emp, dict):
        raise TypeError("emp must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")
    if not changes:
        logger.info(
            "adjust_employee_profile: no changes provided; user_type=%s",
            "admin" if has_admin_rights else "regular",
        )
        return dict(emp)

    if has_admin_rights:
        updated = dict(emp)
        updated.update(changes)
        logger.info(
            "adjust_employee_profile: updated by admin; fields=%s",
            sorted(changes.keys()),
        )
        return updated

    allowed_changes = {k: v for k, v in changes.items() if k in EDITABLE_FIELDS}
    disallowed_fields = sorted(set(changes.keys()) - EDITABLE_FIELDS)
    if disallowed_fields:
        logger.warning(
            "adjust_employee_profile: regular user attempted restricted fields: %s",
            disallowed_fields,
        )
    if not allowed_changes:
        logger.info("adjust_employee_profile: no changes applied for regular user")
        return dict(emp)

    updated = dict(emp)
    updated.update(allowed_changes)
    logger.info(
        "adjust_employee_profile: updated by regular user; fields=%s",
        sorted(allowed_changes.keys()),
    )
    return updated
