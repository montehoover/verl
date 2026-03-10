EDITABLE_FIELDS = {"name", "position", "department"}


def adjust_employee_details(person: dict, alterations: dict, has_admin_rights: bool) -> dict:
    """
    Modify fields in an employee's profile according to permissions.

    Args:
        person: The current employee profile dictionary.
        alterations: A dictionary of proposed changes (field -> new value).
        has_admin_rights: True if the caller is an admin; otherwise False.

    Returns:
        A new dictionary representing the updated employee profile.
    """
    # Work on a copy to avoid mutating the input profile.
    updated_profile = dict(person)

    if has_admin_rights:
        # Admins can modify any existing field in the profile.
        for field, value in alterations.items():
            if field in updated_profile:
                updated_profile[field] = value
    else:
        # Non-admins can only modify fields listed in EDITABLE_FIELDS.
        for field, value in alterations.items():
            if field in EDITABLE_FIELDS and field in updated_profile:
                updated_profile[field] = value

    return updated_profile
