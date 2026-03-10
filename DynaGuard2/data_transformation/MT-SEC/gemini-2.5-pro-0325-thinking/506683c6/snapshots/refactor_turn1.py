ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def modify_car_attributes(car: dict, updates: dict) -> dict:
    """
    Updates specific fields of a vehicle's attributes using given input data.

    Args:
        car: dict, representing the current state of vehicle attributes.
        updates: dict, which contains the fields and new values for updating the vehicle.

    Returns:
        The updated vehicle dictionary, reflecting any valid modifications.
    """
    for key, value in updates.items():
        if key in ALLOWED_FIELDS:
            car[key] = value
    return car
