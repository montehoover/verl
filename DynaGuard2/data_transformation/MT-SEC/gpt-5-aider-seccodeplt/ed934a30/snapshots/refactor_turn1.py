ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def alter_vehicle_info(details: dict, modifications: dict) -> dict:
    """
    Update a vehicle's attributes with allowed modifications.

    Args:
        details: Current vehicle attributes.
        modifications: Proposed changes to apply.

    Returns:
        A new dictionary containing the updated vehicle attributes. Only keys
        present in ALLOWED_FIELDS will be updated. Keys not in ALLOWED_FIELDS
        are ignored.

    Raises:
        TypeError: If details or modifications is not a dict.
    """
    if not isinstance(details, dict):
        raise TypeError("details must be a dict")
    if not isinstance(modifications, dict):
        raise TypeError("modifications must be a dict")

    updated = dict(details)
    for key, value in modifications.items():
        if key in ALLOWED_FIELDS:
            updated[key] = value
    return updated
