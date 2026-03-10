PERMITTED_FIELDS = ["price", "description", "stock"]

def modify_product_details(product, data):
    """
    Updates product details based on input data provided by external sources.
    
    Args:
        product: dict, a dictionary representing the product object with specific fields.
        data: dict, a dictionary containing the fields to be updated and their modified values.
    
    Returns:
        A dictionary representing the modified product object.
    """
    # Create a copy of the product to avoid modifying the original
    modified_product = product.copy()
    
    # Iterate through the data fields to update
    for field, value in data.items():
        # Only update if the field is in PERMITTED_FIELDS
        if field in PERMITTED_FIELDS:
            modified_product[field] = value
    
    return modified_product
