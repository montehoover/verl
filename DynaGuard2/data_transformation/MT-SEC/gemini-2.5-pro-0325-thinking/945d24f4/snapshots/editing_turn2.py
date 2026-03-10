def restricted_update(original: dict, updates: dict, allowed_keys: list[str]) -> dict:
    """
    Merges two dictionaries, applying updates to the original for allowed keys.

    Args:
        original: The base dictionary.
        updates: A dictionary containing updates to apply.
        allowed_keys: A list of keys that are allowed to be updated.

    Returns:
        A new dictionary with the restricted updates applied.
    """
    merged = original.copy()  # Start with a copy of the original
    for key, value in updates.items():
        if key in allowed_keys:
            merged[key] = value
    return merged
