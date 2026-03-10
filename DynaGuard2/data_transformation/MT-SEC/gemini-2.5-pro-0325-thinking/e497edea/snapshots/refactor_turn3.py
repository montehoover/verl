import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def _is_field_updatable(field: str) -> bool:
    """Checks if a field is allowed to be updated."""
    return field in ALLOWED_FIELDS

def _update_vehicle_attribute(auto: dict, field: str, value: any) -> None:
    """Updates a specific attribute of the vehicle."""
    original_value = auto.get(field)
    auto[field] = value
    logging.info(f"Attribute '{field}' changed from '{original_value}' to '{value}'")

def adjust_vehicle_info(auto: dict, modifications: dict) -> dict:
    """
    Modifies specific attributes of a vehicle object based on provided inputs.

    Args:
        auto: A dictionary object representing the vehicle with its attributes.
        modifications: A dictionary of the new values mapped to the attributes
                       that need updating.

    Returns:
        The updated vehicle dictionary reflecting proper authorized modifications.
    """
    for field, value in modifications.items():
        if _is_field_updatable(field):
            _update_vehicle_attribute(auto, field, value)
    return auto
