PERMITTED_FIELDS = ['price', 'description', 'stock']


def update_product(product, updates):
    """
    Update product details with the provided updates.
    
    Args:
        product (dict): The original product dictionary
        updates (dict): Dictionary containing keys and values to update
        
    Returns:
        dict: The updated product dictionary
    """
    # Create a copy of the product to avoid modifying the original
    updated_product = product.copy()
    
    # Update the product with all the updates
    updated_product.update(updates)
    
    return updated_product


def restricted_update(product, updates, modifiable_fields):
    """
    Update product details with the provided updates, but only for allowed fields.
    
    Args:
        product (dict): The original product dictionary
        updates (dict): Dictionary containing keys and values to update
        modifiable_fields (list): List of field names that are allowed to be modified
        
    Returns:
        dict: The updated product dictionary with only allowed fields modified
    """
    # Create a copy of the product to avoid modifying the original
    updated_product = product.copy()
    
    # Only update fields that are in the modifiable_fields list
    for field, value in updates.items():
        if field in modifiable_fields:
            updated_product[field] = value
    
    return updated_product


def edit_product_info(item, details):
    """
    Edit product information with specific constraints.
    
    Args:
        item (dict): A dictionary of product details
        details (dict): A dictionary of fields and values for update
        
    Returns:
        dict: The updated product dictionary with only permitted fields modified
    """
    # Create a copy of the item to avoid modifying the original
    updated_item = item.copy()
    
    # Only update fields that are in PERMITTED_FIELDS
    for field, value in details.items():
        if field in PERMITTED_FIELDS:
            updated_item[field] = value
    
    return updated_item
