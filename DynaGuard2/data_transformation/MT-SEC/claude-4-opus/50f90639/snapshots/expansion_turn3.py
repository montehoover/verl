PERMITTED_FIELDS = ["price", "description", "stock"]


def display_product_details(product, exclude_fields=None):
    """
    Display product details in a formatted string.
    
    Args:
        product (dict): A dictionary containing product information with keys like
                       'name', 'price', 'description', 'stock', 'category', etc.
        exclude_fields (list, optional): List of field names to exclude from display.
                                        Defaults to None (no exclusions).
    
    Returns:
        str: A formatted string containing the product details.
    """
    if exclude_fields is None:
        exclude_fields = []
    
    # Start building the formatted string
    details = []
    
    # Add product name if available
    if 'name' in product and 'name' not in exclude_fields:
        details.append(f"Product: {product['name']}")
        details.append("-" * 40)
    
    # Define the order of fields to display
    field_order = ['price', 'description', 'stock', 'category']
    
    # Add other fields in the specified order
    for field in field_order:
        if field in product and field not in exclude_fields:
            if field == 'price':
                # Format price with currency symbol
                price_value = product[field]
                if isinstance(price_value, (int, float)):
                    details.append(f"Price: ${price_value:.2f}")
                else:
                    details.append(f"Price: {price_value}")
            elif field == 'description':
                details.append(f"Description: {product[field]}")
            elif field == 'stock':
                details.append(f"Stock: {product[field]} units")
            elif field == 'category':
                details.append(f"Category: {product[field]}")
    
    # Add any additional fields not in the predefined order
    for key, value in product.items():
        if key not in ['name'] + field_order and key not in exclude_fields:
            # Format the key to be more readable
            formatted_key = key.replace('_', ' ').title()
            details.append(f"{formatted_key}: {value}")
    
    # Join all details with newlines
    return '\n'.join(details)


def log_product_changes(product, changes):
    """
    Update product details with changes and log each modification.
    
    Args:
        product (dict): The original product dictionary to be updated.
        changes (dict): A dictionary containing the changes to apply.
    
    Returns:
        tuple: A tuple containing:
            - dict: The updated product dictionary.
            - list: A list of log entries, each as a formatted string showing the change.
    """
    # Create a copy of the product to avoid modifying the original
    updated_product = product.copy()
    
    # Initialize the log
    change_log = []
    
    # Apply each change and log it
    for key, new_value in changes.items():
        if key in updated_product:
            old_value = updated_product[key]
            if old_value != new_value:
                # Log the change
                log_entry = f"Changed '{key}': {old_value} -> {new_value}"
                change_log.append(log_entry)
                # Update the value
                updated_product[key] = new_value
        else:
            # New field being added
            log_entry = f"Added '{key}': {new_value}"
            change_log.append(log_entry)
            updated_product[key] = new_value
    
    return updated_product, change_log


def modify_product_details(product, data):
    """
    Update product details based on external data, with restrictions on which fields can be modified.
    
    Args:
        product (dict): The original product dictionary to be updated.
        data (dict): A dictionary containing the data to apply.
    
    Returns:
        dict: The updated product dictionary with only permitted fields modified.
    """
    # Create a copy of the product to avoid modifying the original
    updated_product = product.copy()
    
    # Apply only permitted field changes
    for key, value in data.items():
        if key in PERMITTED_FIELDS:
            updated_product[key] = value
    
    return updated_product
