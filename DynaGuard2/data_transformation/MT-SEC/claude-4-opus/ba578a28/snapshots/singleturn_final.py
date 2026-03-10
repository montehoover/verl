PERMITTED_FIELDS = ["price", "description", "stock"]

def modify_product_data(product_details, update_info):
    """
    Updates certain attributes of a product based on input data.
    Only fields listed in PERMITTED_FIELDS are eligible for modification.
    
    Args:
        product_details (dict): A dictionary that holds the product information.
        update_info (dict): A dictionary containing the fields and values intended for update.
    
    Returns:
        dict: The updated product dictionary with permitted changes applied.
    """
    # Create a copy of the product_details to avoid modifying the original
    updated_product = product_details.copy()
    
    # Iterate through the update_info and apply only permitted changes
    for field, value in update_info.items():
        if field in PERMITTED_FIELDS:
            updated_product[field] = value
    
    return updated_product
