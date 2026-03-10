def merge_dicts(original, updates):
    """Merge two dictionaries, with updates overriding original values."""
    result = original.copy()
    result.update(updates)
    return result

def restricted_update(original, updates, allowed_fields):
    """Update dictionary with only allowed fields from updates."""
    result = original.copy()
    for field in allowed_fields:
        if field in updates:
            result[field] = updates[field]
    return result
