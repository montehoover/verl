ALLOWED_FIELDS = ['make', 'model', 'year', 'registration']

def modify_car_attributes(car: dict, updates: dict) -> dict:
    """
    Modifies car attributes based on a dictionary of updates, restricted to allowed fields.

    Args:
        car: A dictionary representing the car's attributes.
        updates: A dictionary with proposed modifications.

    Returns:
        The updated car dictionary.
    """
    updated_car = car.copy()
    for key, value in updates.items():
        if key in ALLOWED_FIELDS:
            updated_car[key] = value
    return updated_car
