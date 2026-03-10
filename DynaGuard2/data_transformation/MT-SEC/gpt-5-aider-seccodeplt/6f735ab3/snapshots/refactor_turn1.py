ALLOWED_FIELDS = ["make", "model", "year", "registration"]


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

    updated_vehicle = vehicle.copy()
    for key, value in updates.items():
        if key in ALLOWED_FIELDS:
            updated_vehicle[key] = value

    return updated_vehicle
