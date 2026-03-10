def merge_dicts(original: dict, new_data: dict) -> dict:
    """
    Updates a dictionary with new values.

    Args:
        original: The original dictionary.
        new_data: A dictionary with keys and values to update in the original.

    Returns:
        The updated dictionary.
    """
    updated_dict = original.copy()
    updated_dict.update(new_data)
    return updated_dict
