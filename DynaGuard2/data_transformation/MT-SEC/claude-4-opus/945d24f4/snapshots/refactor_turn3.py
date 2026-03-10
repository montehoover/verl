import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def is_field_allowed(field, allowed_fields):
    """
    Check if a field is allowed to be updated.
    
    Args:
        field (str): The field name to check.
        allowed_fields (list): List of allowed field names.
        
    Returns:
        bool: True if the field is allowed, False otherwise.
    """
    return field in allowed_fields


def apply_field_updates(car_details, changes, allowed_fields):
    """
    Apply updates to car details for allowed fields only.
    
    Args:
        car_details (dict): Original car details dictionary.
        changes (dict): Dictionary of fields to update with new values.
        allowed_fields (list): List of allowed field names.
        
    Returns:
        dict: Updated car details dictionary.
    """
    updated_car = car_details.copy()
    
    for field, value in changes.items():
        if is_field_allowed(field, allowed_fields):
            old_value = updated_car.get(field, "N/A")
            updated_car[field] = value
            logger.info(f"Updated field '{field}': '{old_value}' -> '{value}'")
        else:
            logger.warning(f"Attempted to update disallowed field: '{field}'")
    
    return updated_car


def update_car_information(car_details, changes):
    """
    Update car information with allowed changes.
    
    Args:
        car_details (dict): A dictionary representing the vehicle's details.
        changes (dict): Dictionary containing the fields to be updated and the new values.
        
    Returns:
        dict: A dictionary reflecting the updates after valid modifications have been applied.
    """
    logger.info(f"Starting car information update with {len(changes)} potential changes")
    updated_car = apply_field_updates(car_details, changes, ALLOWED_FIELDS)
    logger.info("Car information update completed")
    
    return updated_car
