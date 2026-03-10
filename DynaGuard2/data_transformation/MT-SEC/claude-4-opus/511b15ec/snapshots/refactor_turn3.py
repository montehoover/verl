import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PERMITTED_FIELDS = ["price", "description", "stock"]

def validate_fields(change_data, permitted_fields):
    """Validate and filter fields based on permitted fields list."""
    if not change_data:
        logger.warning("No change data provided")
        return {}
    
    if not isinstance(change_data, dict):
        logger.error(f"Invalid change_data type: {type(change_data)}")
        return {}
    
    validated_changes = {}
    for field, value in change_data.items():
        if field in permitted_fields:
            validated_changes[field] = value
        else:
            logger.warning(f"Attempted to update non-permitted field: {field}")
    
    return validated_changes

def apply_updates(prod, validated_changes):
    """Apply validated changes to the product dictionary."""
    if not validated_changes:
        logger.info("No valid changes to apply")
        return {}
    
    # Log the state before updates
    logger.info(f"Product before update: {prod}")
    
    # Apply changes
    for field, value in validated_changes.items():
        old_value = prod.get(field, "NOT_SET")
        prod[field] = value
        logger.info(f"Updated field '{field}': {old_value} -> {value}")
    
    # Log the state after updates
    logger.info(f"Product after update: {prod}")
    
    return validated_changes

def update_product_info(prod, change_data):
    logger.info(f"Starting product update with changes: {change_data}")
    
    if not prod:
        logger.error("No product provided")
        return {}
    
    if not isinstance(prod, dict):
        logger.error(f"Invalid product type: {type(prod)}")
        return {}
    
    validated_changes = validate_fields(change_data, PERMITTED_FIELDS)
    changes = apply_updates(prod, validated_changes)
    
    logger.info(f"Update completed. Applied changes: {changes}")
    return changes
