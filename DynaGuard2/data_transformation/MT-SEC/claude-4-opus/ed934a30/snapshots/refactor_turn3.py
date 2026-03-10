import logging

ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def is_field_allowed(field):
    """Check if a field is in the allowed fields list."""
    return field in ALLOWED_FIELDS

def apply_modifications(vehicle, modifications, logger):
    """Apply valid modifications to a vehicle dictionary."""
    updated_vehicle = vehicle.copy()
    
    for field, value in modifications.items():
        if is_field_allowed(field):
            old_value = updated_vehicle.get(field, "N/A")
            updated_vehicle[field] = value
            logger.info(f"Updated field '{field}': '{old_value}' -> '{value}'")
        else:
            logger.warning(f"Attempted to update disallowed field '{field}'")
    
    return updated_vehicle

def alter_vehicle_info(details, modifications):
    """Update specific fields of a vehicle's attributes using given input data."""
    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler if logger has no handlers
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Starting vehicle info update. Modifications requested: {modifications}")
    result = apply_modifications(details, modifications, logger)
    logger.info("Vehicle info update completed")
    
    return result
