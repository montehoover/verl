PERMITTED_FIELDS = ["price", "description", "stock"]


def display_product_info(product_info, exclude_fields=None):
    """
    Display product information in a formatted string.
    
    Args:
        product_info (dict): Dictionary containing product attributes
        exclude_fields (list, optional): List of fields to exclude from display
    
    Returns:
        str: Formatted string with product details
    """
    if exclude_fields is None:
        exclude_fields = []
    
    # Define the order and labels for display
    field_mapping = {
        'name': 'Product Name',
        'price': 'Price',
        'description': 'Description',
        'stock': 'Stock Quantity',
        'category': 'Category'
    }
    
    lines = []
    lines.append("=" * 40)
    lines.append("PRODUCT INFORMATION")
    lines.append("=" * 40)
    
    for field, label in field_mapping.items():
        if field in product_info and field not in exclude_fields:
            value = product_info[field]
            if field == 'price':
                value = f"${value:.2f}"
            lines.append(f"{label}: {value}")
    
    lines.append("=" * 40)
    
    return "\n".join(lines)


def log_product_changes(product_info, changes):
    """
    Update product details with changes and log each modification.
    
    Args:
        product_info (dict): Original product information
        changes (dict): Dictionary of changes to apply
    
    Returns:
        tuple: (updated_product, change_log)
            - updated_product: Dictionary with updated values
            - change_log: List of change descriptions
    """
    updated_product = product_info.copy()
    change_log = []
    
    for field, new_value in changes.items():
        if field in updated_product:
            old_value = updated_product[field]
            if old_value != new_value:
                updated_product[field] = new_value
                change_log.append(f"{field}: {old_value} -> {new_value}")
        else:
            updated_product[field] = new_value
            change_log.append(f"{field}: [NEW] -> {new_value}")
    
    return updated_product, change_log


def update_item_information(product_info, new_data):
    """
    Update product information with new data, only allowing updates to permitted fields.
    
    Args:
        product_info (dict): Original product information
        new_data (dict): Dictionary of new data to apply
    
    Returns:
        dict: Updated product dictionary
    """
    updated_product = product_info.copy()
    
    for field, new_value in new_data.items():
        if field in PERMITTED_FIELDS:
            updated_product[field] = new_value
    
    return updated_product
