def update_dict_values(data: dict, updates: dict) -> dict:
    """
    Updates values in a dictionary with values from another dictionary.

    Args:
        data: The original dictionary.
        updates: A dictionary containing keys and values to update in the data.

    Returns:
        A new dictionary with the updated key-value pairs.
    """
    updated_data = data.copy()
    updated_data.update(updates)
    return updated_data


EDITABLE_FIELDS = {'name', 'position', 'department'}


def edit_profile_data(profile: dict, modifications: dict, is_superuser: bool) -> dict:
    """
    Edits an employee profile based on modifications and user permissions.

    Args:
        profile: Dictionary of employee details.
        modifications: Dictionary of proposed changes.
        is_superuser: Boolean indicating if the user has superuser permissions.

    Returns:
        The revised employee profile dictionary.
    """
    revised_profile = profile.copy()
    for key, value in modifications.items():
        if is_superuser or key in EDITABLE_FIELDS:
            revised_profile[key] = value
    return revised_profile
