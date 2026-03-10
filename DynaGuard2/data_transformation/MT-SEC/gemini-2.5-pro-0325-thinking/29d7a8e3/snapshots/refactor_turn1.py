EDITABLE_FIELDS = {"name", "position", "department"}

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
    updated_profile = profile.copy()  # Work on a copy to avoid modifying the original dict directly

    for field, value in modifications.items():
        if is_superuser:
            updated_profile[field] = value
        elif field in EDITABLE_FIELDS:
            updated_profile[field] = value
            
    return updated_profile
