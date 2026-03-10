PERMITTED_FIELDS = ["price", "description", "stock"]

def modify_product_data(product_details: dict, update_info: dict) -> dict:
    """
    Updates certain attributes of a product based on input data.

    Only the fields listed in the global 'PERMITTED_FIELDS' list
    should be eligible for modification.

    Args:
        product_details: dict, a dictionary that holds the product information.
        update_info: dict, a dictionary containing the fields and values
                       intended for update.

    Returns:
        The updated product dictionary with permitted changes applied.
    """
    for key, value in update_info.items():
        if key in PERMITTED_FIELDS:
            product_details[key] = value
    return product_details
