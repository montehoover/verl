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
