ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def modify_car_details(car: dict, changes: dict) -> dict:
    """
    Modify specific attributes of a vehicle dictionary based on provided inputs.

    Only fields listed in ALLOWED_FIELDS are updated. Fields not listed are ignored.
    The original car dictionary is not mutated; a new updated dictionary is returned.

    Args:
        car: dict representing the vehicle and its attributes.
        changes: dict mapping attribute names to new values.

    Returns:
        dict: Updated vehicle dictionary reflecting authorized modifications.
    """
    if not isinstance(car, dict):
        raise TypeError("car must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    updated_car = dict(car)  # shallow copy to avoid mutating the input

    for field, value in changes.items():
        if field in ALLOWED_FIELDS:
            updated_car[field] = value

    return updated_car
