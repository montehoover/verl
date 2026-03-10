import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PERMITTED_FIELDS = ["price", "description", "stock"]

def is_field_permitted(field):
    """Check if a field is in the permitted fields list."""
    return field in PERMITTED_FIELDS

def apply_field_update(product_details, field, value):
    """Apply a single field update to the product."""
    product_details[field] = value
    return product_details

def filter_permitted_updates(update_info):
    """Filter update_info to only include permitted fields."""
    return {field: value for field, value in update_info.items() if is_field_permitted(field)}

def modify_product_data(product_details, update_info):
    # Log original product details
    logger.info(f"Original product details: {product_details}")
    
    # Log update information
    logger.info(f"Update information received: {update_info}")
    
    # Filter and log permitted updates
    permitted_updates = filter_permitted_updates(update_info)
    if permitted_updates:
        logger.info(f"Permitted updates to be applied: {permitted_updates}")
    else:
        logger.info("No permitted updates found")
    
    # Apply updates
    for field, value in permitted_updates.items():
        apply_field_update(product_details, field, value)
        logger.info(f"Updated field '{field}' to '{value}'")
    
    # Log final product details
    logger.info(f"Final product details: {product_details}")
    
    return product_details
