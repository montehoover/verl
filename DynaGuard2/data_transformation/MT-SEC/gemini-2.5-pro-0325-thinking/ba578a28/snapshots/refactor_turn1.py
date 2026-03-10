PERMITTED_FIELDS = ["price", "description", "stock"]

def modify_product_data(product_details: dict, update_info: dict) -> dict:
    """
    Updates certain attributes of a product based on input data.

    Only the fields listed in the global 'PERMITTED_FIELDS' list
    are eligible for modification.

    Args:
        product_details: A dictionary that holds the product information.
        update_info: A dictionary containing the fields and values
                     intended for update.

    Returns:
        The updated product dictionary with permitted changes applied.
    """
    for field, value in update_info.items():
        if field in PERMITTED_FIELDS:
            product_details[field] = value
    return product_details
