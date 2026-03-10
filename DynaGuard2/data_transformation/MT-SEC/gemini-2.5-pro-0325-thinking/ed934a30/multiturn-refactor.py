import logging

ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def _is_field_allowed(field_name: str, allowed_fields_list: list) -> bool:
    """Checks if a field is in the list of allowed fields."""
    return field_name in allowed_fields_list


def _apply_valid_modifications(current_details: dict, modifications_to_apply: dict, allowed_fields_list: list) -> dict:
    """Applies valid modifications to a copy of the vehicle details."""
    updated_vehicle_details = current_details.copy()
    for key, value in modifications_to_apply.items():
        if _is_field_allowed(key, allowed_fields_list):
            old_value = updated_vehicle_details.get(key)
            updated_vehicle_details[key] = value
            logging.info(f"Field '{key}' updated from '{old_value}' to '{value}'")
    return updated_vehicle_details


def alter_vehicle_info(details: dict, modifications: dict) -> dict:
    """
    Updates specific fields of a vehicle's attributes using given input data.

    Args:
        details: dict, representing the current state of vehicle attributes.
        modifications: dict, which contains the fields and new values for updating the vehicle.

    Returns:
        The updated vehicle dictionary, reflecting any valid modifications.
    """
    # Configure logging
    # In a real application, this might be configured once at the application's entry point.
    # For this exercise, configuring it here as requested.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    return _apply_valid_modifications(details, modifications, ALLOWED_FIELDS)
