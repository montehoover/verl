ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def is_allowed_field(field: str) -> bool:
    """
    Check whether a given field name is allowed to be updated.

    Args:
        field: The field name to check.

    Returns:
        True if the field is in ALLOWED_FIELDS; otherwise, False.
    """
    return field in ALLOWED_FIELDS


def apply_vehicle_updates(vehicle: dict, allowed_updates: dict) -> dict:
    """
    Return a new vehicle dictionary with allowed updates applied.

    This function does not mutate the input vehicle.

    Args:
        vehicle: The original vehicle dictionary.
        allowed_updates: A dictionary of updates that are already known to be allowed.

    Returns:
        A new dictionary representing the updated vehicle object.
    """
    updated = vehicle.copy()
    updated.update(allowed_updates)
    return updated


def update_vehicle_info(vehicle: dict, updates: dict) -> dict:
    """
    Update allowed attributes of a vehicle dictionary based on provided updates.

    Only fields listed in the global ALLOWED_FIELDS will be updated.
    The original vehicle dictionary is not mutated; a new updated dict is returned.

    Args:
        vehicle: A dictionary representing the vehicle (e.g., with keys like 'make', 'model', 'year', 'owner', 'registration').
        updates: A dictionary of fields to update and their new values.

    Returns:
        A new dictionary representing the updated vehicle object.

    Raises:
        TypeError: If either vehicle or updates is not a dictionary.
    """
    if not isinstance(vehicle, dict):
        raise TypeError("vehicle must be a dict")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dict")

    allowed_updates = {k: v for k, v in updates.items() if is_allowed_field(k)}
    return apply_vehicle_updates(vehicle, allowed_updates)
