ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def adjust_vehicle_info(auto: dict, modifications: dict) -> dict:
    """
    Modifies specific attributes of a vehicle object based on provided inputs.

    Args:
        auto: A dictionary object representing the vehicle with its attributes.
        modifications: A dictionary of the new values mapped to the attributes
                       that need updating.

    Returns:
        The updated vehicle dictionary reflecting proper authorized modifications.
    """
    for key, value in modifications.items():
        if key in ALLOWED_FIELDS:
            auto[key] = value
    return auto
