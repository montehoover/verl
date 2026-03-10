PERMITTED_FIELDS = ["price", "description", "stock"]

def update_item_information(product_info, new_data):
    """
    Modifies certain product attributes based on information from a provided data dictionary.
    
    Args:
        product_info (dict): A dictionary instance representing a product.
        new_data (dict): A dictionary containing key-value pairs representing intended updates to the product details.
    
    Returns:
        dict: A dictionary object reflecting the updated product state.
    """
    # Create a copy of the product_info to avoid modifying the original
    updated_product = product_info.copy()
    
    # Iterate through the new_data and update only permitted fields
    for field, value in new_data.items():
        if field in PERMITTED_FIELDS:
            updated_product[field] = value
    
    return updated_product
