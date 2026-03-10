ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def _is_field_allowed(field: str, allowed_fields_list: list[str]) -> bool:
    """Checks if a field is allowed to be updated."""
    return field in allowed_fields_list


def _apply_vehicle_update(vehicle_data: dict, field_to_update: str, new_value: any) -> None:
    """Applies a single update to the vehicle data."""
    vehicle_data[field_to_update] = new_value


def update_vehicle_info(vehicle: dict, updates: dict) -> dict:
    """
    Updates vehicle attributes based on input data.

    Only attributes specified in the global ALLOWED_FIELDS list can be modified.

    Args:
        vehicle: A dictionary representing the vehicle object.
        updates: A dictionary containing the attributes to be updated
                 and their new values.

    Returns:
        A dictionary representing the updated vehicle object.
    """
    for key, value in updates.items():
        if _is_field_allowed(key, ALLOWED_FIELDS):
            _apply_vehicle_update(vehicle, key, value)
    return vehicle
