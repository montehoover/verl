PERMITTED_FIELDS = ['price', 'description', 'stock']

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

def amend_product_features(item, payload):
    """
    Update product features with constraints.
    
    Args:
        item: Dictionary representing the product
        payload: Dictionary with new values for fields to update
        
    Returns:
        Dictionary with updated product information
    """
    updated_item = item.copy()
    for field, value in payload.items():
        if field in PERMITTED_FIELDS:
            updated_item[field] = value
    return updated_item
