def merge_dicts(original, updates):
    """
    Return a new dictionary with updates applied to the original.
    This performs a shallow merge: keys in 'updates' override those in 'original'.
    Inputs are not mutated.
    """
    if not isinstance(original, dict):
        raise TypeError("original must be a dict")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dict")

    merged = original.copy()
    merged.update(updates)
    return merged
