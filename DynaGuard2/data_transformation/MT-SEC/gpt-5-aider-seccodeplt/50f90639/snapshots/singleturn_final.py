PERMITTED_FIELDS = ["price", "description", "stock"]

def modify_product_details(product: dict, data: dict) -> dict:
    """
    Update a product dictionary with values from `data`, limited to PERMITTED_FIELDS.

    Args:
        product (dict): A dictionary representing the product object with fields
                        such as 'price', 'description', 'stock', 'category', etc.
        data (dict): A dictionary containing the fields to be updated and their new values.

    Returns:
        dict: A new dictionary representing the modified product object.
    """
    if not isinstance(product, dict):
        raise TypeError("product must be a dict")
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    # Start from a copy to avoid mutating the original product
    updated = product.copy()

    # Apply only permitted updates
    for field, value in data.items():
        if field in PERMITTED_FIELDS:
            updated[field] = value

    return updated
