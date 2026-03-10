def selective_update(product, new_data, allowed_fields):
    """
    Update product dictionary with new data, but only for allowed fields.
    
    Args:
        product: Dictionary containing product information
        new_data: Dictionary with keys and values to update in product
        allowed_fields: List of field names that are allowed to be updated
        
    Returns:
        Updated product dictionary
    """
    updated_product = product.copy()
    for field, value in new_data.items():
        if field in allowed_fields:
            updated_product[field] = value
    return updated_product
