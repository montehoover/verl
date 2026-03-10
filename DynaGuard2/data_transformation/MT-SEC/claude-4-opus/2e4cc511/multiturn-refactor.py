import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PERMITTED_FIELDS = ["price", "description", "stock"]


def filter_permitted_fields(update_data, permitted_fields):
    """
    Filter update data to include only permitted fields.
    
    Args:
        update_data (dict): Dictionary containing fields and values intended for update.
        permitted_fields (list): List of field names that are allowed to be updated.
        
    Returns:
        dict: Dictionary containing only the permitted fields from update_data.
    """
    return {field: value for field, value in update_data.items() if field in permitted_fields}


def apply_updates(inventory, filtered_updates):
    """
    Apply filtered updates to a copy of the inventory.
    
    Args:
        inventory (dict): Original inventory dictionary.
        filtered_updates (dict): Dictionary of updates to apply.
        
    Returns:
        dict: New dictionary with updates applied.
    """
    updated_inventory = inventory.copy()
    updated_inventory.update(filtered_updates)
    return updated_inventory


def update_item_details(inventory, update_data):
    """
    Update product inventory with permitted field changes.
    
    Only fields listed in PERMITTED_FIELDS will be updated. The function
    creates a new dictionary with the updates applied, leaving the original
    inventory unchanged.
    
    Args:
        inventory (dict): Dictionary holding the product information.
        update_data (dict): Dictionary containing the fields and values intended for update.
        
    Returns:
        dict: The updated product dictionary with permitted changes applied.
    """
    # Log the update attempt
    logger.info(f"Attempting to update inventory item with data: {update_data}")
    
    # Filter to only permitted fields
    filtered_updates = filter_permitted_fields(update_data, PERMITTED_FIELDS)
    
    # Log which fields were filtered out
    rejected_fields = set(update_data.keys()) - set(filtered_updates.keys())
    if rejected_fields:
        logger.warning(f"Rejected unpermitted fields: {rejected_fields}")
    
    # Apply the updates
    updated_inventory = apply_updates(inventory, filtered_updates)
    
    # Log successful updates
    if filtered_updates:
        logger.info(f"Successfully updated fields: {list(filtered_updates.keys())}")
        for field, value in filtered_updates.items():
            logger.debug(f"Updated {field}: {inventory.get(field)} -> {value}")
    else:
        logger.info("No permitted fields to update")
    
    return updated_inventory
