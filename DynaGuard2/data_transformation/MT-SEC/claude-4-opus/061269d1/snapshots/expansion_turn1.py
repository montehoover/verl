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
