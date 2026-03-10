EDITABLE_FIELDS = {"name", "position", "department"}

def validate_modifications(modifications):
    """
    Normalize and validate the modifications input.

    Returns:
        A dictionary of modifications (empty dict if None provided).
    """
    if modifications is None:
        return {}
    if not isinstance(modifications, dict):
        # Keep behavior minimal: accept only dict-like inputs
        # to avoid surprising mutations or errors later.
        raise ValueError("modifications must be a dict or None")
    # Return a shallow copy to preserve purity
    return dict(modifications)


def select_allowed_edits(modifications: dict, is_superuser: bool, profile: dict) -> dict:
    """
    Select which modifications are allowed based on privilege level and existing profile keys.
    Admins can edit any existing key in the profile.
    Non-admins can only edit keys in EDITABLE_FIELDS that also exist in the profile.
    """
    profile_keys = set(profile.keys())

    if is_superuser:
        return {k: v for k, v in modifications.items() if k in profile_keys}

    allowed_keys = EDITABLE_FIELDS & profile_keys
    return {k: v for k, v in modifications.items() if k in allowed_keys}


def apply_edited_fields(profile: dict, edits: dict) -> dict:
    """
    Apply allowed edits to a copy of the profile and return the updated profile.
    """
    updated = dict(profile)
    if edits:
        updated.update(edits)
    return updated


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

    normalized_mods = validate_modifications(modifications)
    allowed_edits = select_allowed_edits(normalized_mods, is_superuser, profile)
    return apply_edited_fields(profile, allowed_edits)
