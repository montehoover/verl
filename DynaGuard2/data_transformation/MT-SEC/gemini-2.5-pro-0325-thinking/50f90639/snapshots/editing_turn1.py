def update_values(original: dict, new_data: dict) -> dict:
    """
    Updates the values in the original dictionary with values from new_data.

    Args:
        original: The dictionary to be updated.
        new_data: A dictionary containing keys and values to update in the original.

    Returns:
        The updated dictionary.
    """
    original.update(new_data)
    return original
