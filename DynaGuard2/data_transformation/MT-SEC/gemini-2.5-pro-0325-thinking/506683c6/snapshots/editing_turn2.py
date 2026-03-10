def restricted_update(original: dict, new_data: dict, allowed_fields: list[str]) -> dict:
    """
    Updates a dictionary with new data, restricted to allowed fields.

    Args:
        original: The original dictionary.
        new_data: A dictionary with keys and values to update in the original.
        allowed_fields: A list of keys that are allowed to be updated.

    Returns:
        The updated dictionary.
    """
    updated_dict = original.copy()
    for key, value in new_data.items():
        if key in allowed_fields:
            updated_dict[key] = value
    return updated_dict
