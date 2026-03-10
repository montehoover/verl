PERMITTED_FIELDS = ['price', 'description', 'stock']

def modify_product_data(product_details: dict, update_info: dict) -> dict:
    """
    Modifies product data for permitted fields only.

    Args:
        product_details: A dictionary of product information.
        update_info: A dictionary of fields and values for update.

    Returns:
        The updated product dictionary.
    """
    for key, value in update_info.items():
        if key in PERMITTED_FIELDS:
            product_details[key] = value
    return product_details
