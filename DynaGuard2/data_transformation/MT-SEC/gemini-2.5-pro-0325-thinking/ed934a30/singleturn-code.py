ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def alter_vehicle_info(details: dict, modifications: dict) -> dict:
    """
    Updates specific fields of a vehicle's attributes using given input data.

    Args:
        details: dict, representing the current state of vehicle attributes.
        modifications: dict, which contains the fields and new values for updating the vehicle.

    Returns:
        The updated vehicle dictionary, reflecting any valid modifications.
    """
    for key, value in modifications.items():
        if key in ALLOWED_FIELDS:
            details[key] = value
    return details
