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
