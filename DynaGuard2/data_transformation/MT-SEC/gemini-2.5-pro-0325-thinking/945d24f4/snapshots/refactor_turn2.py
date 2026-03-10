ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def _is_field_allowed(field: str, allowed_fields_list: list) -> bool:
    """
    Checks if a specific field is allowed for modification.

    Args:
        field: The name of the field to check.
        allowed_fields_list: A list of field names that are allowed.

    Returns:
        True if the field is allowed, False otherwise.
    """
    return field in allowed_fields_list

def _apply_validated_updates(car_details: dict, changes: dict, allowed_fields_list: list) -> dict:
    """
    Applies updates to a copy of car_details for fields that are allowed.

    Args:
        car_details: The original dictionary of car details.
        changes: A dictionary of proposed changes (field: new_value).
        allowed_fields_list: A list of field names that are allowed for updates.

    Returns:
        A new dictionary with the validated updates applied.
    """
    updated_details = car_details.copy()
    for key, value in changes.items():
        if _is_field_allowed(key, allowed_fields_list):
            updated_details[key] = value
    return updated_details

def update_car_information(car_details: dict, changes: dict) -> dict:
    """
    Modifies specific fields of a vehicle's details.

    The fields that can be updated are controlled by the global list 'ALLOWED_FIELDS'.

    Args:
        car_details: A dictionary representing the vehicle's details.
        changes: This dictionary contains the fields to be updated and the new values.

    Returns:
        A dictionary reflecting the updates after valid modifications have been applied.
    """
    return _apply_validated_updates(car_details, changes, ALLOWED_FIELDS)
