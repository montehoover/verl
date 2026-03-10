PERMITTED_FIELDS = ["price", "description", "stock"]


def filter_permitted_fields(data: dict, permitted_fields: list) -> dict:
    """
    Return a new dict containing only keys from data that are in permitted_fields.
    """
    return {k: v for k, v in data.items() if k in permitted_fields}


def apply_updates(base: dict, updates: dict) -> dict:
    """
    Return a new dict with updates applied to base without mutating the original base dict.
    """
    result = base.copy()
    result.update(updates)
    return result


def update_item_information(product_info: dict, new_data: dict) -> dict:
    """
    Update product information with values from new_data for keys permitted by PERMITTED_FIELDS.

    Args:
        product_info: Existing product dictionary (e.g., contains 'price', 'description', 'stock', 'category', etc.).
        new_data: Dictionary of proposed updates.

    Returns:
        A new dictionary reflecting the updated product state.
    """
    permitted_updates = filter_permitted_fields(new_data, PERMITTED_FIELDS)
    updated_product = apply_updates(product_info, permitted_updates)
    return updated_product
