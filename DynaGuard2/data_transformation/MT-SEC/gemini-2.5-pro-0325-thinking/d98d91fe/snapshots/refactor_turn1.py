ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def modify_car_details(car: dict, changes: dict) -> dict:
    """
    Modifies specific attributes of a vehicle object based on provided inputs.

    Args:
        car: dict, a dictionary object representing the vehicle with its attributes.
        changes: dict, a dictionary of the new values mapped to the attributes
                 that need updating.

    Returns:
        The updated vehicle dictionary reflecting proper authorized modifications.
    """
    updated_car = car.copy()  # Work on a copy to avoid modifying the original dict directly
    for key, value in changes.items():
        if key in ALLOWED_FIELDS:
            updated_car[key] = value
    return updated_car
