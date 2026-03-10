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
