PERMITTED_FIELDS = ["price", "description", "stock"]

def amend_product_features(item, payload):
    """
    Update certain fields of a product using incoming data.
    Only fields present in the globally defined 'PERMITTED_FIELDS' list can be modified.
    
    Args:
        item (dict): The dictionary representing the product object with its corresponding fields.
        payload (dict): A dictionary containing the new values for the fields that need to be updated.
    
    Returns:
        dict: A dictionary reflecting the changes made to the product object.
    """
    # Create a copy of the item to avoid modifying the original
    updated_item = item.copy()
    
    # Iterate through the payload and update only permitted fields
    for field, value in payload.items():
        if field in PERMITTED_FIELDS:
            updated_item[field] = value
    
    return updated_item
