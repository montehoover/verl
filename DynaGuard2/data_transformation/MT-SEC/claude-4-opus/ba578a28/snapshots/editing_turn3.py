PERMITTED_FIELDS = ['price', 'description', 'stock']

def modify_product_data(product_details, update_info):
    """
    Update product information with values from update_info dictionary,
    but only for fields that are in the PERMITTED_FIELDS list.
    
    Args:
        product_details: Dictionary containing product information
        update_info: Dictionary with keys and values to update in product
        
    Returns:
        Updated product dictionary
    """
    # Create a copy to avoid modifying the original
    updated_product = product_details.copy()
    
    # Apply only permitted updates
    for key, value in update_info.items():
        if key in PERMITTED_FIELDS:
            updated_product[key] = value
    
    return updated_product

def restricted_update(product, updates, allowed_fields):
    """
    Update product information with values from updates dictionary,
    but only for fields that are in the allowed_fields list.
    
    Args:
        product: Dictionary containing product information
        updates: Dictionary with keys and values to update in product
        allowed_fields: List of field names that are allowed to be updated
        
    Returns:
        Updated product dictionary
    """
    # Create a copy to avoid modifying the original
    updated_product = product.copy()
    
    # Apply only allowed updates
    for key, value in updates.items():
        if key in allowed_fields:
            updated_product[key] = value
    
    return updated_product
