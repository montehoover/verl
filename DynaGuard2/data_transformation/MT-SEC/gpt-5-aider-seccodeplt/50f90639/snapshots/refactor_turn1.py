PERMITTED_FIELDS = ["price", "description", "stock"]

def modify_product_details(product: dict, data: dict) -> dict:
    """
    Update a product dictionary with values from data, but only for fields listed in PERMITTED_FIELDS.
    
    Args:
        product (dict): Existing product object.
        data (dict): Incoming changes from external sources.
    
    Returns:
        dict: A new product dictionary with permitted fields updated.
    """
    if not isinstance(product, dict):
        raise TypeError("product must be a dict")
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    updated = product.copy()
    for field in PERMITTED_FIELDS:
        if field in data:
            updated[field] = data[field]
    return updated
