def merge_dicts(original, new_data):
    original.update(new_data)
    return original

def validate_required_fields(data, required_keys):
    return required_keys.issubset(data.keys())
