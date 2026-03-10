ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def update_car_information(car_details: dict, changes: dict) -> dict:
    """
    Update allowed fields of a vehicle's details.

    Parameters:
        car_details (dict): The original vehicle details.
        changes (dict): Fields to update with new values.

    Returns:
        dict: A new dictionary reflecting updates applied only to allowed fields.

    Notes:
        - Only keys present in ALLOWED_FIELDS are updated.
        - The original car_details dict is not mutated.
    """
    if not isinstance(car_details, dict):
        raise TypeError("car_details must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    updated = car_details.copy()
    for key, value in changes.items():
        if key in ALLOWED_FIELDS:
            updated[key] = value

    return updated
