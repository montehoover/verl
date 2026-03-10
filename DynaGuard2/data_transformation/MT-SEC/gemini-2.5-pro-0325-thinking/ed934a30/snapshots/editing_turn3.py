ALLOWED_FIELDS = ['make', 'model', 'year', 'registration']

def alter_vehicle_info(details: dict, modifications: dict) -> dict:
    """
    Updates vehicle information, restricted to allowed fields.

    Args:
        details: A dictionary of vehicle attributes.
        modifications: A dictionary of proposed changes.

    Returns:
        A new dictionary with the updated vehicle information.
    """
    updated_details = details.copy()
    for key, value in modifications.items():
        if key in ALLOWED_FIELDS:
            updated_details[key] = value
    return updated_details
