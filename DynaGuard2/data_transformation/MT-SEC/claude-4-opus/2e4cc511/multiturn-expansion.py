import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PERMITTED_FIELDS = ["price", "description", "stock"]

def get_product_details(product, fields_to_exclude=None):
    """
    Retrieve and format product details.
    
    Args:
        product (dict): Product dictionary with attributes like price, description, stock, category
        fields_to_exclude (list, optional): List of field names to exclude from the output
    
    Returns:
        str: Formatted string with product details
    """
    if fields_to_exclude is None:
        fields_to_exclude = []
    
    details = []
    
    # Define the order and formatting for common fields
    field_formatters = {
        'name': lambda v: f"Product: {v}",
        'price': lambda v: f"Price: ${v:.2f}" if isinstance(v, (int, float)) else f"Price: {v}",
        'description': lambda v: f"Description: {v}",
        'stock': lambda v: f"Stock: {v} units",
        'category': lambda v: f"Category: {v}",
    }
    
    # Process fields in defined order first
    for field, formatter in field_formatters.items():
        if field in product and field not in fields_to_exclude:
            details.append(formatter(product[field]))
    
    # Add any additional fields not in the predefined list
    for key, value in product.items():
        if key not in field_formatters and key not in fields_to_exclude:
            # Format the key nicely (capitalize and replace underscores)
            formatted_key = key.replace('_', ' ').title()
            details.append(f"{formatted_key}: {value}")
    
    return '\n'.join(details)


def log_product_changes(product, changes):
    """
    Log changes made to product details and update the product.
    
    Args:
        product (dict): Original product dictionary
        changes (dict): Dictionary containing fields to update with new values
    
    Returns:
        dict: Updated product dictionary
    """
    product_id = product.get('id', 'Unknown')
    product_name = product.get('name', 'Unknown Product')
    
    for field, new_value in changes.items():
        if field in product:
            original_value = product[field]
            if original_value != new_value:
                logging.info(
                    f"Product '{product_name}' (ID: {product_id}) - "
                    f"Changed '{field}' from '{original_value}' to '{new_value}'"
                )
                product[field] = new_value
        else:
            logging.info(
                f"Product '{product_name}' (ID: {product_id}) - "
                f"Added new field '{field}' with value '{new_value}'"
            )
            product[field] = new_value
    
    # Add timestamp for last modification
    product['last_modified'] = datetime.now().isoformat()
    
    return product


def update_item_details(inventory, update_data):
    """
    Update product details while respecting permitted field constraints.
    
    Args:
        inventory (dict): Product dictionary to update
        update_data (dict): Dictionary containing fields to update with new values
    
    Returns:
        dict: Updated product dictionary with only permitted changes applied
    """
    # Filter update_data to only include permitted fields
    permitted_updates = {
        field: value 
        for field, value in update_data.items() 
        if field in PERMITTED_FIELDS
    }
    
    # Apply the permitted updates using the existing log_product_changes function
    updated_inventory = log_product_changes(inventory, permitted_updates)
    
    return updated_inventory
