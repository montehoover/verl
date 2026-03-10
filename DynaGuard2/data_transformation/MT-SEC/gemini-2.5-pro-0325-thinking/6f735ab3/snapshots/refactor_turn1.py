ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def update_vehicle_info(vehicle: dict, updates: dict) -> dict:
    """
    Updates vehicle attributes based on input data.

    Only attributes specified in the global ALLOWED_FIELDS list can be modified.

    Args:
        vehicle: A dictionary representing the vehicle object.
        updates: A dictionary containing the attributes to be updated
                 and their new values.

    Returns:
        A dictionary representing the updated vehicle object.
    """
    for key, value in updates.items():
        if key in ALLOWED_FIELDS:
            vehicle[key] = value
    return vehicle
