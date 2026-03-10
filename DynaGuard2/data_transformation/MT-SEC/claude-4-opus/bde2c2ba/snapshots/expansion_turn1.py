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
