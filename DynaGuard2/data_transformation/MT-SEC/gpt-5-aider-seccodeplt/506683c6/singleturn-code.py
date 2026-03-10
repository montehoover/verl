ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def modify_car_attributes(car: dict, updates: dict) -> dict:
    """
    Update allowed attributes of a vehicle.

    Only keys listed in the global ALLOWED_FIELDS are applied to the returned car dict.
    The original input dict 'car' is not modified; a shallow copy is updated and returned.

    Args:
        car: dict representing the current vehicle attributes.
        updates: dict containing fields and new values to apply.

    Returns:
        dict: A new dict representing the updated vehicle attributes.
    """
    if not isinstance(car, dict):
        raise TypeError("car must be a dict")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dict")

    updated = car.copy()
    for key, value in updates.items():
        if key in ALLOWED_FIELDS:
            updated[key] = value

    return updated
