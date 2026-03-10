PERMITTED_FIELDS = ["price", "description", "stock"]


def update_item_information(product_info: dict, new_data: dict) -> dict:
    """
    Update product information with values from new_data for keys permitted by PERMITTED_FIELDS.
    
    Args:
        product_info: Existing product dictionary (e.g., contains 'price', 'description', 'stock', 'category', etc.).
        new_data: Dictionary of proposed updates.

    Returns:
        A new dictionary reflecting the updated product state.
    """
    updated = product_info.copy()
    for key, value in new_data.items():
        if key in PERMITTED_FIELDS:
            updated[key] = value
    return updated
