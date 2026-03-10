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
    updated_details = details.copy()  # Work on a copy to avoid modifying the original dict directly
    for key, value in modifications.items():
        if key in ALLOWED_FIELDS:
            updated_details[key] = value
    return updated_details
