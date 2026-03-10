ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def alter_vehicle_info(details: dict, modifications: dict) -> dict:
    """
    Update specific fields of a vehicle's attributes using given input data,
    constrained by the global ALLOWED_FIELDS list.

    Args:
        details: dict representing the current state of vehicle attributes.
        modifications: dict containing fields and new values for updating.

    Returns:
        dict: The updated vehicle dictionary reflecting any valid modifications.
    """
    if not isinstance(details, dict) or not isinstance(modifications, dict):
        raise TypeError("details and modifications must be dictionaries")

    updated = details.copy()

    for key, value in modifications.items():
        if key in ALLOWED_FIELDS:
            updated[key] = value

    return updated
