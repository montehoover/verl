PERMITTED_FIELDS = ["price", "description", "stock"]

def update_item_information(product_info: dict, new_data: dict) -> dict:
    """
    Modifies product attributes based on information from a provided data dictionary.
    Updates are permitted only for the fields present in the global list PERMITTED_FIELDS.

    Args:
        product_info: A dictionary instance representing a product.
        new_data: A dictionary containing key-value pairs representing intended
                  updates to the product details.

    Returns:
        A dictionary object reflecting the updated product state.
    """
    updated_product_info = product_info.copy()  # Avoid modifying the original dict directly
    for key, value in new_data.items():
        if key in PERMITTED_FIELDS:
            updated_product_info[key] = value
    return updated_product_info
