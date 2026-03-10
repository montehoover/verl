def merge_dicts(original, new_data):
    """
    Update 'original' with the contents of 'new_data' and return the updated dictionary.
    This is a shallow merge and mutates 'original'.
    """
    original.update(new_data)
    return original
