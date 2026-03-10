PERMITTED_FIELDS = ["price", "description", "stock"]

def modify_product_details(product: dict, data: dict) -> dict:
    """
    Updates product details based on input data provided by external sources.

    Args:
        product: dict, a dictionary representing the product object with specific fields.
        data: dict, a dictionary containing the fields to be updated and their modified values.

    Returns:
        A dictionary representing the modified product object.
    """
    updated_product = product.copy()  # Avoid modifying the original product dict directly
    for key, value in data.items():
        if key in PERMITTED_FIELDS:
            updated_product[key] = value
    return updated_product
