"""Utilities to modify staff information with permission checks."""

EDITABLE_FIELDS = {"name", "position", "department"}


def _validate_input_types(staff: dict, changes: dict) -> None:
    """
    Validate that provided staff and changes are dictionaries.

    Raises:
        TypeError: If staff or changes is not a dict.
    """
    if not isinstance(staff, dict):
        raise TypeError("staff must be a dict")

    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")


def _filter_allowed_changes(
    changes: dict,
    admin_status: bool,
    editable_fields: set,
) -> dict:
    """
    Determine which changes are permitted based on admin status.

    Args:
        changes (dict): Proposed updates as key-value pairs.
        admin_status (bool): True if the requester is an admin.
        editable_fields (set): Fields non-admin users are allowed to edit.

    Returns:
        dict: Changes that are allowed to be applied.
    """
    if not changes:
        return {}

    if admin_status:
        return dict(changes)

    return {key: value for key, value in changes.items() if key in editable_fields}


def _apply_allowed_changes(staff: dict, allowed_changes: dict) -> dict:
    """
    Return a new staff dictionary with allowed changes applied.

    Args:
        staff (dict): Current employee information.
        allowed_changes (dict): Validated, permitted changes.

    Returns:
        dict: Updated staff dictionary.
    """
    updated = staff.copy()

    if not allowed_changes:
        return updated

    updated.update(allowed_changes)
    return updated


def modify_staff_info(staff: dict, changes: dict, admin_status: bool) -> dict:
    """
    Modify an employee's information with respect to admin permissions.

    Args:
        staff (dict): Current employee information.
        changes (dict): Proposed updates as key-value pairs.
        admin_status (bool): True if the requester is an admin; otherwise False.

    Returns:
        dict: A new dictionary with valid updates applied.
    """
    _validate_input_types(staff, changes)

    if not changes:
        return staff.copy()

    allowed_changes = _filter_allowed_changes(
        changes=changes,
        admin_status=admin_status,
        editable_fields=EDITABLE_FIELDS,
    )

    return _apply_allowed_changes(staff=staff, allowed_changes=allowed_changes)
