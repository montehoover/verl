def merge_dicts(original, new_data):
    """
    Update 'original' with the contents of 'new_data' and return the updated dictionary.
    This is a shallow merge and mutates 'original'.
    """
    original.update(new_data)
    return original

def validate_required_fields(data, required_keys):
    """
    Return True if all keys in 'required_keys' are present in 'data'; otherwise False.
    """
    return required_keys.issubset(data.keys())
