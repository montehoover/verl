EDITABLE_FIELDS = {"name", "position", "department"}


def edit_profile_data(profile: dict, modifications: dict, is_superuser: bool) -> dict:
    """
    Modify fields within an employee's profile according to permissions.

    Args:
        profile (dict): The current employee profile with details such as
                        name, position, salary, department, social_security_number, etc.
        modifications (dict): Fields and new values to update in the profile.
        is_superuser (bool): If True, all fields can be updated. If False, only
                             fields listed in EDITABLE_FIELDS can be updated.

    Returns:
        dict: The revised employee profile dictionary after applying updates.
    """
    if profile is None:
        raise TypeError("profile must be a dict, got None")
    if modifications is None:
        # Nothing to apply; return a copy to avoid mutating the input.
        return dict(profile)

    if not isinstance(profile, dict):
        raise TypeError(f"profile must be a dict, got {type(profile).__name__}")
    if not isinstance(modifications, dict):
        raise TypeError(f"modifications must be a dict, got {type(modifications).__name__}")

    # Work on a shallow copy to avoid mutating the provided profile.
    updated_profile = dict(profile)

    if is_superuser:
        # Admins can update any provided fields (including adding new keys).
        for key, value in modifications.items():
            updated_profile[key] = value
        return updated_profile

    # Non-admins can update only fields listed in EDITABLE_FIELDS.
    for key, value in modifications.items():
        if key in EDITABLE_FIELDS:
            updated_profile[key] = value

    return updated_profile
