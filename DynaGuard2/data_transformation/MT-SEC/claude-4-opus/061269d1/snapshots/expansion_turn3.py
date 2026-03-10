import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PERMITTED_FIELDS = ["price", "description", "stock"]

def display_product_info(item, fields_to_exclude=None):
    """
    Display product information in a formatted string.
    
    Args:
        item (dict): Product dictionary with properties like price, description, stock, category
        fields_to_exclude (list, optional): List of field names to exclude from display
    
    Returns:
        str: Formatted product information
    """
    if fields_to_exclude is None:
        fields_to_exclude = []
    
    # Define field labels for better display
    field_labels = {
        'name': 'Product Name',
        'price': 'Price',
        'description': 'Description',
        'stock': 'Stock Available',
        'category': 'Category',
        'sku': 'SKU',
        'brand': 'Brand'
    }
    
    lines = []
    lines.append("=" * 40)
    lines.append("PRODUCT INFORMATION")
    lines.append("=" * 40)
    
    for field, value in item.items():
        if field not in fields_to_exclude:
            label = field_labels.get(field, field.replace('_', ' ').title())
            
            # Format price with currency symbol
            if field == 'price':
                formatted_value = f"${value:.2f}"
            elif field == 'stock':
                formatted_value = f"{value} units"
            else:
                formatted_value = str(value)
            
            lines.append(f"{label}: {formatted_value}")
    
    lines.append("=" * 40)
    
    return '\n'.join(lines)


def log_product_changes(item, details):
    """
    Log changes made to product attributes and update the product.
    
    Args:
        item (dict): Current product dictionary
        details (dict): Dictionary with proposed changes
    
    Returns:
        dict: Updated product dictionary
    """
    # Create a copy of the item to avoid modifying the original
    updated_item = item.copy()
    
    # Get product identifier for logging
    product_id = item.get('sku', item.get('id', 'Unknown'))
    
    # Track if any changes were made
    changes_made = False
    
    for field, new_value in details.items():
        if field in updated_item:
            old_value = updated_item[field]
            
            # Only log if the value actually changed
            if old_value != new_value:
                changes_made = True
                
                # Log the change
                logging.info(
                    f"Product [{product_id}] - Field '{field}' changed: "
                    f"'{old_value}' -> '{new_value}'"
                )
                
                # Update the product
                updated_item[field] = new_value
        else:
            # Field doesn't exist in original product, add it
            changes_made = True
            
            logging.info(
                f"Product [{product_id}] - New field '{field}' added: '{new_value}'"
            )
            
            updated_item[field] = new_value
    
    if not changes_made:
        logging.info(f"Product [{product_id}] - No changes were made")
    else:
        # Log summary of changes
        logging.info(
            f"Product [{product_id}] - Update completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
    return updated_item


def edit_product_info(item, details):
    """
    Update product information while adhering to specific field constraints.
    
    Args:
        item (dict): Current product dictionary
        details (dict): Dictionary with proposed changes
    
    Returns:
        dict: Updated product dictionary with permitted changes applied
    """
    # Create a copy of the item to avoid modifying the original
    updated_item = item.copy()
    
    # Filter details to only include permitted fields
    permitted_changes = {
        field: value 
        for field, value in details.items() 
        if field in PERMITTED_FIELDS
    }
    
    # Apply permitted changes
    for field, new_value in permitted_changes.items():
        if field in updated_item:
            updated_item[field] = new_value
    
    return updated_item
