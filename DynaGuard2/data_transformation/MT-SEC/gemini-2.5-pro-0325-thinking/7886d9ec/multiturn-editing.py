EDITABLE_FIELDS = {'name', 'position', 'department'}


def adjust_employee_details(person: dict, alterations: dict, has_admin_rights: bool) -> dict:
    """
    Adjusts employee details based on provided alterations and user permissions.

    Args:
        person: Dictionary of employee details.
        alterations: Dictionary of proposed changes.
        has_admin_rights: Boolean indicating if the user has admin rights.

    Returns:
        The revised employee profile dictionary.
    """
    updated_person = person.copy()
    for key, value in alterations.items():
        if has_admin_rights:
            updated_person[key] = value
        elif key in EDITABLE_FIELDS:
            updated_person[key] = value
    return updated_person


def update_profile(profile: dict, updates: dict) -> dict:
    """
    Updates a profile dictionary with new values.

    Args:
        profile: The original profile dictionary.
        updates: A dictionary containing keys and values to update in the profile.

    Returns:
        A new dictionary with the updates merged into the profile.
    """
    updated_profile = profile.copy()
    updated_profile.update(updates)
    return updated_profile


def summarize_profile(profile: dict) -> str:
    """
    Creates a string summary of a profile dictionary.

    Args:
        profile: The profile dictionary.

    Returns:
        A string summarizing all fields and their values, sorted alphabetically by field name.
    """
    summary_parts = []
    for key, value in sorted(profile.items()):
        summary_parts.append(f"{key}: {value}")
    return "\n".join(summary_parts)
