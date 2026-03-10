def update_dict_entries(original: dict, updates: dict) -> dict:
    """
    Update 'original' with key/value pairs from 'updates' and return the updated dictionary.

    This function mutates 'original' in place and also returns it.
    """
    original.update(updates)
    return original
