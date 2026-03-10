PERMITTED_FIELDS = ["price", "description", "stock"]

def edit_product_info(item: dict, details: dict) -> dict:
    """
    Updates certain attributes of a product based on input data.

    Only the fields listed in the global 'PERMITTED_FIELDS' list
    are eligible for modification.

    Args:
        item: dict, a dictionary that holds the product information.
        details: dict, a dictionary containing the fields and values
                 intended for update.

    Returns:
        The updated product dictionary with permitted changes applied.
    """
    for key, value in details.items():
        if key in PERMITTED_FIELDS:
            item[key] = value
    return item
