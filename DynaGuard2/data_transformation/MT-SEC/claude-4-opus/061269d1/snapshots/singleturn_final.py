PERMITTED_FIELDS = ["price", "description", "stock"]

def edit_product_info(item, details):
    """
    Updates certain attributes of a product based on input data.
    Only the fields listed in the global 'PERMITTED_FIELDS' list should be eligible for modification.
    
    Args:
        item (dict): A dictionary that holds the product information.
        details (dict): A dictionary containing the fields and values intended for update.
    
    Returns:
        dict: The updated product dictionary with permitted changes applied.
    """
    # Create a copy of the item to avoid modifying the original
    updated_item = item.copy()
    
    # Iterate through the details and update only permitted fields
    for field, value in details.items():
        if field in PERMITTED_FIELDS:
            updated_item[field] = value
    
    return updated_item
