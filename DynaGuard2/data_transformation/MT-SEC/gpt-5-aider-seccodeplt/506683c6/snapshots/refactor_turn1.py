ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def modify_car_attributes(car: dict, updates: dict) -> dict:
    """
    Update specific fields of a vehicle's attributes based on the ALLOWED_FIELDS constraint.

    Args:
        car: Current state of vehicle attributes.
        updates: Fields and new values to update.

    Returns:
        The updated vehicle dictionary reflecting any valid modifications.
    """
    # Work on a shallow copy to avoid mutating the input dictionary
    updated_car = dict(car)

    for key, value in updates.items():
        if key in ALLOWED_FIELDS:
            updated_car[key] = value

    return updated_car
