PERMITTED_FIELDS = ["price", "description", "stock"]


def update_item_information(product_info: dict, new_data: dict) -> dict:
    """
    Update a product's information based on provided new_data, but only for fields
    listed in PERMITTED_FIELDS. Returns a new dictionary reflecting the updated product
    state and does not mutate the input product_info.

    Args:
        product_info (dict): The original product dictionary (e.g., contains keys like
            'price', 'description', 'stock', 'category', etc.).
        new_data (dict): A dictionary with intended updates.

    Returns:
        dict: A new dictionary with permitted fields updated and all other fields preserved.
    """
    # Start with a shallow copy so we don't mutate the original input
    updated_product = dict(product_info) if product_info is not None else {}

    if not new_data:
        return updated_product

    # Apply only permitted updates
    for key in PERMITTED_FIELDS:
        if key in new_data:
            updated_product[key] = new_data[key]

    return updated_product
