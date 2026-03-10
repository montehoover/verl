def update_record(data: dict, modifications: dict) -> dict:
    """
    Updates a dictionary with key-value pairs from another dictionary.

    Args:
        data: The original dictionary.
        modifications: A dictionary containing keys and values to update in data.

    Returns:
        The updated dictionary.
    """
    data.update(modifications)
    return data
