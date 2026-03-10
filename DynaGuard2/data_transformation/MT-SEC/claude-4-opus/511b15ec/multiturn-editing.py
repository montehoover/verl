PERMITTED_FIELDS = ["price", "description", "stock"]

def update_product_info(prod: dict, change_data: dict) -> dict:
    """
    Updates product information for permitted fields only.
    
    Args:
        prod: dict - The product dictionary with fields like price, description, etc.
        change_data: dict - Dictionary containing new values to update
        
    Returns:
        dict - Dictionary with updated product information for permitted fields only
    """
    # Create a copy of the product dictionary to avoid modifying the original
    updated_product = prod.copy()
    
    # Process each field in change_data
    for field, new_value in change_data.items():
        # Only update if field is in PERMITTED_FIELDS
        if field in PERMITTED_FIELDS:
            updated_product[field] = new_value
    
    return updated_product
