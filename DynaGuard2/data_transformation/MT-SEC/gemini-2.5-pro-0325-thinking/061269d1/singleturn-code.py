PERMITTED_FIELDS = ["price", "description", "stock"]

def edit_product_info(item: dict, details: dict) -> dict:
    """
    Updates certain attributes of a product based on input data.

    Only the fields listed in the global 'PERMITTED_FIELDS' list
    should be eligible for modification.

    Args:
        item: dict, a dictionary that holds the product information.
        details: dict, a dictionary containing the fields and values
                 intended for update.

    Returns:
        The updated product dictionary with permitted changes applied.
    """
    updated_item = item.copy()  # Work on a copy to avoid modifying the original dict directly
    for key, value in details.items():
        if key in PERMITTED_FIELDS:
            updated_item[key] = value
    return updated_item
