def restricted_update(data: dict, modifications: dict, has_permission: bool) -> dict:
    """
    Updates a dictionary with new values from another dictionary,
    respecting field permissions.

    Args:
        data: The original dictionary.
        modifications: A dictionary containing keys and values to update in data.
        has_permission: If True, all fields can be updated.
                        If False, only allowed_fields can be updated.

    Returns:
        The updated dictionary.
    """
    allowed_fields = {'name', 'description'}  # Predefined set of allowed fields

    if has_permission:
        data.update(modifications)
    else:
        for key, value in modifications.items():
            if key in allowed_fields:
                data[key] = value
    return data
