def merge_dicts(original, updates):
    """
    Return a new dictionary with updates applied to the original.

    This function performs a shallow merge:
    - It does not mutate either input dictionary.
    - Keys present in `updates` overwrite or add to those in `original`.
    - All fields are allowed to be updated, including setting values to None.
    """
    if original is None:
        original = {}
    if updates is None:
        updates = {}

    if not isinstance(original, dict):
        raise TypeError("original must be a dict")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dict")

    result = original.copy()
    result.update(updates)
    return result
