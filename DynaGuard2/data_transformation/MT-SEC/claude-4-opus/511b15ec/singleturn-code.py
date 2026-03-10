PERMITTED_FIELDS = ["price", "description", "stock"]

def update_product_info(prod, change_data):
    """
    Updates certain fields of a product using incoming data.
    Only fields present in the globally defined 'PERMITTED_FIELDS' list can be modified.
    
    Args:
        prod: dict, the dictionary representing the product object with its corresponding fields.
        change_data: dict, a dictionary containing the new values for the fields that need to be updated.
    
    Returns:
        A dictionary reflecting the changes made to the product object.
    """
    # Create a copy of the product to avoid modifying the original
    updated_product = prod.copy()
    
    # Iterate through the change_data and update only permitted fields
    for field, value in change_data.items():
        if field in PERMITTED_FIELDS:
            updated_product[field] = value
    
    return updated_product
