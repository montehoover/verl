PERMITTED_FIELDS = ['price', 'description', 'stock']

def update_item_details(inventory, update_data):
    """
    Update product details in inventory with constraints on which fields can be modified.
    
    Args:
        inventory: Dictionary of product information
        update_data: Dictionary of fields and values for update
    
    Returns:
        Updated product dictionary with only permitted fields modified
    """
    updated_inventory = inventory.copy()
    for field, value in update_data.items():
        if field in PERMITTED_FIELDS:
            updated_inventory[field] = value
    return updated_inventory

def restricted_update(product, new_data, allowed_fields):
    """
    Update product details by merging new_data into product dictionary,
    but only for fields that are in the allowed_fields list.
    
    Args:
        product: Dictionary containing product details
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
