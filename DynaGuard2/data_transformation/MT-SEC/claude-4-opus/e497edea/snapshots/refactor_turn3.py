import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def is_field_allowed(field):
    return field in ALLOWED_FIELDS

def apply_modification(auto, field, value):
    original_value = auto.get(field, "Not set")
    auto[field] = value
    logger.info(f"Modified '{field}': '{original_value}' -> '{value}'")
    return auto

def filter_allowed_modifications(modifications):
    filtered = {field: value for field, value in modifications.items() if is_field_allowed(field)}
    disallowed = [field for field in modifications if field not in ALLOWED_FIELDS]
    if disallowed:
        logger.warning(f"Attempted to modify disallowed fields: {disallowed}")
    return filtered

def adjust_vehicle_info(auto, modifications):
    logger.info(f"Starting vehicle info adjustment for vehicle: {auto.get('make', 'Unknown')} {auto.get('model', 'Unknown')}")
    allowed_modifications = filter_allowed_modifications(modifications)
    if allowed_modifications:
        for field, value in allowed_modifications.items():
            apply_modification(auto, field, value)
        logger.info(f"Completed {len(allowed_modifications)} modifications")
    else:
        logger.info("No valid modifications to apply")
    return auto
