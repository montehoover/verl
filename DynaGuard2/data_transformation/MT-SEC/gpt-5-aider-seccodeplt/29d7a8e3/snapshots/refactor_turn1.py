EDITABLE_FIELDS = {"name", "position", "department"}

def edit_profile_data(profile: dict, modifications: dict, is_superuser: bool) -> dict:
    """
    Modify an employee profile based on the provided modifications and privilege level.

    Args:
        profile: Current employee profile dictionary.
        modifications: Dictionary of fields and new values to apply.
        is_superuser: Whether the caller has admin privileges.

    Returns:
        A new dictionary representing the updated employee profile.
    """
    if profile is None:
        raise ValueError("profile must be a dict")

    if modifications is None:
        modifications = {}

    # Work on a shallow copy to avoid mutating the original input
    updated = dict(profile)

    if is_superuser:
        # Admins can update any existing field in the profile
        for key, value in modifications.items():
            if key in updated:
                updated[key] = value
        return updated

    # Non-admins can only update fields listed in EDITABLE_FIELDS
    for key, value in modifications.items():
        if key in EDITABLE_FIELDS and key in updated:
            updated[key] = value

    return updated
