def restricted_update(original: dict, updates: dict, allowed_fields: list[str]) -> dict:
    """
    Updates entries in the original dictionary with values from the updates dictionary,
    but only for fields specified in allowed_fields.

    Args:
        original: The dictionary to be updated.
        updates: A dictionary containing keys and values to update in the original.
        allowed_fields: A list of strings representing the keys that are allowed to be updated.

    Returns:
        The updated dictionary.
    """
    for key, value in updates.items():
        if key in allowed_fields:
            original[key] = value
    return original
