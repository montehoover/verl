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
