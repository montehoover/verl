PERMITTED_FIELDS = ["price", "description", "stock"]

def modify_product_data(product_details: dict, update_info: dict) -> dict:
    """
    Update the product details with fields from update_info that are permitted.

    Args:
        product_details (dict): Original product information.
        update_info (dict): Fields and values intended for update.

    Returns:
        dict: A new product dictionary with permitted changes applied.
    """
    if not isinstance(product_details, dict) or not isinstance(update_info, dict):
        raise TypeError("product_details and update_info must be dict instances")

    updated = product_details.copy()

    for field, value in update_info.items():
        if field in PERMITTED_FIELDS:
            updated[field] = value

    return updated
