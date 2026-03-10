import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def is_field_allowed(field, allowed_fields):
    """Check if a field is in the list of allowed fields."""
    return field in allowed_fields

def update_car_field(car, field, value):
    """Update a single field in the car dictionary."""
    original_value = car.get(field, "NOT_SET")
    car[field] = value
    logger.info(f"Updated field '{field}': '{original_value}' -> '{value}'")
    return car

def filter_allowed_updates(updates, allowed_fields):
    """Filter updates to only include allowed fields."""
    filtered = {field: value for field, value in updates.items() if is_field_allowed(field, allowed_fields)}
    
    # Log which fields were filtered out
    filtered_out = set(updates.keys()) - set(filtered.keys())
    if filtered_out:
        logger.warning(f"Filtered out disallowed fields: {filtered_out}")
    
    return filtered

def modify_car_attributes(car, updates):
    logger.info(f"Starting car attribute modification. Current car: {car}")
    logger.info(f"Requested updates: {updates}")
    
    allowed_updates = filter_allowed_updates(updates, ALLOWED_FIELDS)
    
    if allowed_updates:
        logger.info(f"Applying allowed updates: {allowed_updates}")
        for field, value in allowed_updates.items():
            car = update_car_field(car, field, value)
    else:
        logger.info("No valid updates to apply")
    
    logger.info(f"Car attribute modification complete. Updated car: {car}")
    return car
