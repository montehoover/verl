import logging

ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def filter_allowed_updates(updates, allowed_fields):
    return {field: value for field, value in updates.items() if field in allowed_fields}

def apply_updates(vehicle, filtered_updates):
    updated_vehicle = vehicle.copy()
    updated_vehicle.update(filtered_updates)
    return updated_vehicle

def update_vehicle_info(vehicle, updates):
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    filtered_updates = filter_allowed_updates(updates, ALLOWED_FIELDS)
    
    # Log the updates
    for field, new_value in filtered_updates.items():
        if field in vehicle:
            original_value = vehicle[field]
            logger.info(f"Updating {field}: '{original_value}' -> '{new_value}'")
        else:
            logger.info(f"Adding new field {field}: '{new_value}'")
    
    return apply_updates(vehicle, filtered_updates)
