def merge_dicts(original: dict, updates: dict) -> dict:
    """
    Merges two dictionaries, applying updates to the original.

    Args:
        original: The base dictionary.
        updates: A dictionary containing updates to apply.

    Returns:
        A new dictionary with the updates applied.
    """
    merged = original.copy()  # Start with a copy of the original
    merged.update(updates)  # Apply updates
    return merged
