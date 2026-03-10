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
