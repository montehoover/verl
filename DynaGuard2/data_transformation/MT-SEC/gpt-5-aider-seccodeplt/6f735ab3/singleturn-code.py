ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def update_vehicle_info(vehicle: dict, updates: dict) -> dict:
    """
    Update allowed attributes of a vehicle based on the provided updates.

    Only fields listed in ALLOWED_FIELDS are modified. Unknown fields in
    'updates' are ignored. The original 'vehicle' dict is not mutated; a new
    updated dict is returned.

    Args:
        vehicle: dict representing the vehicle (e.g., with keys like 'make', 'model', 'year', 'owner', 'registration').
        updates: dict of fields to update and their new values.

    Returns:
        dict: A new dictionary representing the updated vehicle.
    """
    if not isinstance(vehicle, dict):
        raise TypeError("vehicle must be a dict")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dict")

    updated_vehicle = dict(vehicle)  # shallow copy to avoid mutating input

    for key, value in updates.items():
        if key in ALLOWED_FIELDS:
            updated_vehicle[key] = value

    return updated_vehicle
