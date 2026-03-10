PERMITTED_FIELDS = ['price', 'description', 'stock']


def update_values(original, new_data):
    """
    Update values in a dictionary with new data.
    
    Args:
        original: The original dictionary to update
        new_data: Dictionary containing keys and values to update in the original
        
    Returns:
        The updated dictionary
    """
    # Create a copy of the original to avoid modifying it
    updated = original.copy()
    
    # Update with new data
    updated.update(new_data)
    
    return updated


def restricted_update(original, new_data, allowed_fields):
    """
    Update values in a dictionary with new data, but only for allowed fields.
    
    Args:
        original: The original dictionary to update
        new_data: Dictionary containing keys and values to update in the original
        allowed_fields: List of field names that are allowed to be updated
        
    Returns:
        The updated dictionary with only allowed fields modified
    """
    # Create a copy of the original to avoid modifying it
    updated = original.copy()
    
    # Only update fields that are in the allowed list
    for field, value in new_data.items():
        if field in allowed_fields:
            updated[field] = value
    
    return updated


def modify_product_details(product, data):
    """
    Modify product details with specific constraints.
    
    Args:
        product: A dictionary representing the product
        data: A dictionary with fields to be updated
        
    Returns:
        The modified product dictionary with only permitted fields updated
    """
    # Create a copy of the product to avoid modifying the original
    modified_product = product.copy()
    
    # Only update fields that are in PERMITTED_FIELDS
    for field, value in data.items():
        if field in PERMITTED_FIELDS:
            modified_product[field] = value
    
    return modified_product
