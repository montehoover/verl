import logging

PERMITTED_FIELDS = ["price", "description", "stock"]

def is_field_permitted(field):
    return field in PERMITTED_FIELDS

def update_permitted_fields(item, details):
    updated_item = item.copy()
    for field, value in details.items():
        if is_field_permitted(field):
            updated_item[field] = value
    return updated_item

def edit_product_info(item, details):
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Get the updated item
    updated_item = update_permitted_fields(item, details)
    
    # Log the changes
    for field, value in details.items():
        if is_field_permitted(field) and field in item:
            if item[field] != value:
                logger.info(f"Updated field '{field}' from '{item[field]}' to '{value}'")
        elif is_field_permitted(field) and field not in item:
            logger.info(f"Added new field '{field}' with value '{value}'")
    
    return updated_item
