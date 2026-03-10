def modify_staff_details(employee_data, changes, is_superuser):
    """
    Apply updates to employee data with permission controls.

    - Superusers can update any field.
    - Regular users can update only fields in EDITABLE_FIELDS.

    Args:
        employee_data (dict): The current employee attributes.
        changes (dict): Proposed field updates.
        is_superuser (bool): Whether the acting user has superuser privileges.

    Returns:
        dict: The modified employee data dictionary.
    """
    # Ensure we are working with dictionaries; otherwise, use safe defaults.
    base = dict(employee_data) if isinstance(employee_data, dict) else {}
    if not isinstance(changes, dict):
        return base

    if is_superuser:
        base.update(changes)
        return base

    # Regular users: restrict to allowed editable fields.
    editable = globals().get("EDITABLE_FIELDS", {"name", "position", "department"})
    try:
        editable_set = set(editable)
    except Exception:
        editable_set = {"name", "position", "department"}

    for key, value in changes.items():
        if key in editable_set:
            base[key] = value

    return base
