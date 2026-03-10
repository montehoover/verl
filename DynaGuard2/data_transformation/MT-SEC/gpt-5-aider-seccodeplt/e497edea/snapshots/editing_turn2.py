def restricted_update(original: dict, updates: dict, allowed_fields: list[str]) -> dict:
    """
    Update 'original' with key/value pairs from 'updates' for keys present in 'allowed_fields'.

    Only keys listed in 'allowed_fields' will be updated or added; all other keys in 'updates' are ignored.
    This function mutates 'original' in place and returns it.
    """
    allowed = set(allowed_fields)
    for key, value in updates.items():
        if key in allowed:
            original[key] = value
    return original
