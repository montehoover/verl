ALLOWED_FIELDS = ['make', 'model', 'year', 'registration']

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

def update_car_information(car_details: dict, changes: dict) -> dict:
    """
    Updates car information based on a predefined list of allowed fields.

    Args:
        car_details: A dictionary of vehicle details.
        changes: A dictionary of fields to update.

    Returns:
        A new dictionary with the updates applied for allowed fields.
    """
    updated_details = car_details.copy()  # Start with a copy of the original
    for key, value in changes.items():
        if key in ALLOWED_FIELDS:
            updated_details[key] = value
    return updated_details
