def update_dict_entries(original: dict, updates: dict) -> dict:
    """
    Updates entries in the original dictionary with values from the updates dictionary.

    Args:
        original: The dictionary to be updated.
        updates: A dictionary containing keys and values to update in the original.

    Returns:
        The updated dictionary.
    """
    original.update(updates)
    return original
