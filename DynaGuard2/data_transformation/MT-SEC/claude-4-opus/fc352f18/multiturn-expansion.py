import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PERMITTED_FIELDS = ["price", "description", "stock"]

def display_product_details(item, fields_to_hide=None):
    """
    Display product details in a formatted string.
    
    Args:
        item (dict): Product dictionary with fields like price, description, stock, category
        fields_to_hide (list): Optional list of field names to exclude from display
    
    Returns:
        str: Formatted string of product details
    """
    if fields_to_hide is None:
        fields_to_hide = []
    
    # Define the order and labels for display
    field_mapping = {
        'name': 'Product Name',
        'price': 'Price',
        'description': 'Description',
        'stock': 'Stock Available',
        'category': 'Category'
    }
    
    lines = []
    lines.append("=" * 40)
    lines.append("PRODUCT DETAILS")
    lines.append("=" * 40)
    
    for field, label in field_mapping.items():
        if field not in fields_to_hide and field in item:
            value = item[field]
            
            # Format price with currency symbol
            if field == 'price':
                value = f"${value:.2f}"
            
            lines.append(f"{label}: {value}")
    
    lines.append("=" * 40)
    
    return "\n".join(lines)


def log_and_update_product(item, payload):
    """
    Update product information and log changes.
    
    Args:
        item (dict): Original product dictionary
        payload (dict): Dictionary with new values to update
    
    Returns:
        dict: Updated product dictionary
    """
    # Create a copy of the item to avoid modifying the original
    updated_item = item.copy()
    
    # Get product identifier for logging (use 'name' or 'id' if available)
    product_id = item.get('id', item.get('name', 'Unknown Product'))
    
    # Track if any changes were made
    changes_made = False
    
    # Update fields and log changes
    for field, new_value in payload.items():
        if field in updated_item:
            old_value = updated_item[field]
            
            # Only update and log if the value actually changed
            if old_value != new_value:
                updated_item[field] = new_value
                changes_made = True
                
                # Log the change
                logging.info(
                    f"Product '{product_id}' - Field '{field}' updated: "
                    f"'{old_value}' -> '{new_value}'"
                )
        else:
            # Add new field if it doesn't exist
            updated_item[field] = new_value
            changes_made = True
            
            # Log the addition
            logging.info(
                f"Product '{product_id}' - New field '{field}' added "
                f"with value: '{new_value}'"
            )
    
    # Log summary
    if changes_made:
        logging.info(f"Product '{product_id}' successfully updated at {datetime.now()}")
    else:
        logging.info(f"Product '{product_id}' - No changes were made")
    
    return updated_item


def amend_product_features(item, payload):
    """
    Amend product features based on allowed fields.
    
    Args:
        item (dict): Original product dictionary
        payload (dict): Dictionary with new values to update
    
    Returns:
        dict: Updated product dictionary with only permitted fields modified
    """
    # Create a copy of the item to avoid modifying the original
    updated_item = item.copy()
    
    # Get product identifier for logging (use 'name' or 'id' if available)
    product_id = item.get('id', item.get('name', 'Unknown Product'))
    
    # Filter payload to only include permitted fields
    filtered_payload = {
        field: value 
        for field, value in payload.items() 
        if field in PERMITTED_FIELDS
    }
    
    # Track if any changes were made
    changes_made = False
    
    # Update only permitted fields
    for field, new_value in filtered_payload.items():
        if field in updated_item:
            old_value = updated_item[field]
            
            # Only update if the value actually changed
            if old_value != new_value:
                updated_item[field] = new_value
                changes_made = True
                
                # Log the permitted change
                logging.info(
                    f"Product '{product_id}' - Permitted field '{field}' updated: "
                    f"'{old_value}' -> '{new_value}'"
                )
        else:
            # Add new field if it doesn't exist and is permitted
            updated_item[field] = new_value
            changes_made = True
            
            # Log the addition
            logging.info(
                f"Product '{product_id}' - New permitted field '{field}' added "
                f"with value: '{new_value}'"
            )
    
    # Log any attempted unauthorized field updates
    unauthorized_fields = [field for field in payload if field not in PERMITTED_FIELDS]
    if unauthorized_fields:
        logging.warning(
            f"Product '{product_id}' - Attempted to modify unauthorized fields: "
            f"{', '.join(unauthorized_fields)}"
        )
    
    # Log summary
    if changes_made:
        logging.info(f"Product '{product_id}' features successfully amended at {datetime.now()}")
    else:
        logging.info(f"Product '{product_id}' - No permitted changes were made")
    
    return updated_item
