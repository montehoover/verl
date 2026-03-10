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


def restricted_update(original, updates, allowed_fields):
    """
    Return a new dictionary with restricted updates applied to the original.

    Behavior:
    - Does not mutate either input dictionary.
    - Only keys present in `allowed_fields` will be applied from `updates`.
    - Keys not in `allowed_fields` are ignored.
    - Supports adding new keys if they are present in `allowed_fields`.

    Args:
        original (dict): The base dictionary.
        updates (dict): Proposed updates to apply.
        allowed_fields (list[str]): List of field names that are allowed to be updated.

    Returns:
        dict: A new dictionary with the allowed updates applied.
    """
    if original is None:
        original = {}
    if updates is None:
        updates = {}
    if allowed_fields is None:
        allowed_fields = []

    if not isinstance(original, dict):
        raise TypeError("original must be a dict")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dict")
    if not isinstance(allowed_fields, list):
        raise TypeError("allowed_fields must be a list of strings")

    for field in allowed_fields:
        if not isinstance(field, str):
            raise TypeError("allowed_fields must be a list of strings")

    allowed_set = set(allowed_fields)
    filtered_updates = {k: v for k, v in updates.items() if k in allowed_set}

    result = original.copy()
    result.update(filtered_updates)
    return result


ALLOWED_FIELDS = ['make', 'model', 'year', 'registration']


def update_vehicle_info(vehicle, updates):
    """
    Return a new vehicle dictionary with only allowed fields updated.

    Only the fields defined in ALLOWED_FIELDS will be modified or added.
    Inputs are not mutated.
    """
    return restricted_update(vehicle, updates, ALLOWED_FIELDS)
