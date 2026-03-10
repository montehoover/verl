def update_info(current_info: dict, new_info: dict) -> dict:
    """
    Updates a dictionary with new information.

    Args:
        current_info: The original dictionary.
        new_info: A dictionary containing updates.

    Returns:
        A new dictionary with the merged information.
    """
    updated_info = current_info.copy()
    updated_info.update(new_info)
    return updated_info
