def merge_dicts(original, updates):
    """Merge two dictionaries, with updates overriding original values."""
    result = original.copy()
    result.update(updates)
    return result
