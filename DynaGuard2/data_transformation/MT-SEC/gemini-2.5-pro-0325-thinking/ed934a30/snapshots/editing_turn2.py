def restricted_update(current_info: dict, new_info: dict, allowed_fields: list) -> dict:
    """
    Updates a dictionary with new information, restricted to allowed fields.

    Args:
        current_info: The original dictionary.
        new_info: A dictionary containing updates.
        allowed_fields: A list of keys that are allowed to be updated.

    Returns:
        A new dictionary with the merged information, restricted to allowed fields.
    """
    updated_info = current_info.copy()
    for key, value in new_info.items():
        if key in allowed_fields:
            updated_info[key] = value
    return updated_info
