ALLOWED_FIELDS = ['make', 'model', 'year', 'registration']

def adjust_vehicle_info(auto: dict, modifications: dict) -> dict:
    """
    Updates vehicle information based on allowed fields.

    Args:
        auto: A dictionary representing the vehicle.
        modifications: A dictionary of new values for attributes that need updating.

    Returns:
        The updated vehicle dictionary reflecting proper authorized modifications.
    """
    for key, value in modifications.items():
        if key in ALLOWED_FIELDS:
            auto[key] = value
    return auto
