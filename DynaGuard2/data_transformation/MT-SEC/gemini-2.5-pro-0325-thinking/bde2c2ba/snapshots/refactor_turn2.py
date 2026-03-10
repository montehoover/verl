PERMITTED_FIELDS = ["price", "description", "stock"]

def _filter_permitted_data(new_data: dict, allowed_fields: list) -> dict:
    """
    Filters new_data to include only keys present in allowed_fields.
    """
    return {key: value for key, value in new_data.items() if key in allowed_fields}

def _apply_updates(product_info: dict, data_to_update: dict) -> dict:
    """
    Applies updates from data_to_update to a copy of product_info.
    """
    updated_product_info = product_info.copy()
    updated_product_info.update(data_to_update)
    return updated_product_info

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
    permitted_updates = _filter_permitted_data(new_data, PERMITTED_FIELDS)
    updated_product_info = _apply_updates(product_info, permitted_updates)
    return updated_product_info
