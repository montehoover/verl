ALLOWED_FIELDS = ['make', 'model', 'year', 'registration']


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


def restricted_update(original, updates, allowed_keys):
    """
    Return a new dictionary with only the allowed updates applied to the original.
    - Only keys present in 'allowed_keys' will be updated/added from 'updates'.
    - Inputs are not mutated.
    """
    if not isinstance(original, dict):
        raise TypeError("original must be a dict")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dict")
    try:
        allowed_set = set(allowed_keys)
    except TypeError:
        raise TypeError("allowed_keys must be an iterable of keys")

    filtered_updates = {k: v for k, v in updates.items() if k in allowed_set}

    merged = original.copy()
    merged.update(filtered_updates)
    return merged


def update_car_information(car_details, changes):
    """
    Return a new dictionary of vehicle details with only allowed fields updated.
    - Only keys present in ALLOWED_FIELDS will be updated/added from 'changes'.
    - Inputs are not mutated.
    """
    if not isinstance(car_details, dict):
        raise TypeError("car_details must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    allowed_set = set(ALLOWED_FIELDS)
    filtered_changes = {k: v for k, v in changes.items() if k in allowed_set}

    updated = car_details.copy()
    updated.update(filtered_changes)
    return updated
