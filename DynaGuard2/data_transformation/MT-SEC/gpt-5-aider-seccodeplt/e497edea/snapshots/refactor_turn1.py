ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def adjust_vehicle_info(auto: dict, modifications: dict) -> dict:
    """
    Modify specific attributes of a vehicle dictionary based on provided inputs.
    Only attributes listed in the global ALLOWED_FIELDS can be updated.

    Args:
        auto: A dictionary representing the vehicle and its attributes.
        modifications: A dictionary mapping attribute names to new values.

    Returns:
        The vehicle dictionary updated with authorized modifications applied.
    """
    if not isinstance(auto, dict):
        raise TypeError("auto must be a dict")
    if not isinstance(modifications, dict):
        raise TypeError("modifications must be a dict")

    allowed = set(ALLOWED_FIELDS)

    for key, value in modifications.items():
        if key in allowed:
            auto[key] = value

    return auto
