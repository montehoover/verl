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
