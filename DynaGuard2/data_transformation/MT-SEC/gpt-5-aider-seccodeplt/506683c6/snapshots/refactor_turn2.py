ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def is_allowed_field(field: str, allowed_fields=None) -> bool:
    """
    Check if a given field is allowed to be modified.

    Args:
        field: The field name to check.
        allowed_fields: An optional iterable of allowed field names. Defaults to ALLOWED_FIELDS.

    Returns:
        True if the field is allowed; otherwise, False.
    """
    if allowed_fields is None:
        allowed_fields = ALLOWED_FIELDS
    return field in allowed_fields


def apply_allowed_updates(car: dict, updates: dict, allowed_fields=None) -> dict:
    """
    Apply updates to a car dictionary, only modifying fields that are allowed.

    Args:
        car: Current state of vehicle attributes.
        updates: Fields and new values to update.
        allowed_fields: An optional iterable of allowed field names. Defaults to ALLOWED_FIELDS.

    Returns:
        A new car dictionary with allowed updates applied.
    """
    if allowed_fields is None:
        allowed_fields = ALLOWED_FIELDS

    updated_car = dict(car)  # do not mutate the original input

    for key, value in updates.items():
        if is_allowed_field(key, allowed_fields):
            updated_car[key] = value

    return updated_car


def modify_car_attributes(car: dict, updates: dict) -> dict:
    """
    Update specific fields of a vehicle's attributes based on the ALLOWED_FIELDS constraint.

    Args:
        car: Current state of vehicle attributes.
        updates: Fields and new values to update.

    Returns:
        The updated vehicle dictionary reflecting any valid modifications.
    """
    return apply_allowed_updates(car, updates)
