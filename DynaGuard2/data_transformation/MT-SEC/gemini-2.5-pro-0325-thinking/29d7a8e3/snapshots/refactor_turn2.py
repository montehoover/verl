EDITABLE_FIELDS = {"name", "position", "department"}

def _validate_modifications(modifications: dict, is_superuser: bool) -> dict:
    """
    Validates the modifications based on user privileges.

    Args:
        modifications: dict, a dictionary specifying the fields and respective
                       new values.
        is_superuser: bool, indicates if the user is an admin.

    Returns:
        dict: A dictionary of validated modifications.
    """
    if is_superuser:
        return modifications.copy()  # Superuser can edit any field

    validated_modifications = {}
    for field, value in modifications.items():
        if field in EDITABLE_FIELDS:
            validated_modifications[field] = value
    return validated_modifications

def _apply_modifications(profile: dict, validated_modifications: dict) -> dict:
    """
    Applies validated modifications to the profile.

    Args:
        profile: dict, the current employee profile.
        validated_modifications: dict, the modifications to apply.

    Returns:
        dict: The updated employee profile.
    """
    updated_profile = profile.copy()
    for field, value in validated_modifications.items():
        updated_profile[field] = value
    return updated_profile

def edit_profile_data(profile: dict, modifications: dict, is_superuser: bool) -> dict:
    """
    Modifies certain fields within an employee's profile.

    Args:
        profile: dict, represents the current employee profile.
        modifications: dict, a dictionary specifying the fields and respective
                       new values meant to update the employee profile.
        is_superuser: bool, indicates whether the operation is carried out by
                      an admin or a non-admin user.

    Returns:
        dict: The revised version of the employee profile dictionary after
              the appropriate fields have been updated.
    """
    validated_modifications = _validate_modifications(modifications, is_superuser)
    updated_profile = _apply_modifications(profile, validated_modifications)
    return updated_profile
