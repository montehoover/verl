PERMITTED_FIELDS = ['price', 'description', 'stock']

def selective_update(product: dict, new_data: dict, allowed_fields: list[str]) -> dict:
    """
    Selectively updates the product dictionary with new data for allowed fields.

    Args:
        product: The original product dictionary.
        new_data: A dictionary containing keys and values to update in the product.
        allowed_fields: A list of strings representing the keys that are allowed to be updated.

    Returns:
        The updated product dictionary.
    """
    updated_product = product.copy()
    for key, value in new_data.items():
        if key in allowed_fields:
            updated_product[key] = value
    return updated_product

def amend_product_features(item: dict, payload: dict) -> dict:
    """
    Amends product features based on a payload, restricted to permitted fields.

    Args:
        item: The original product dictionary.
        payload: A dictionary containing new values for fields to update.

    Returns:
        A dictionary reflecting the changes made to the product object.
    """
    updated_item = item.copy()
    for key, value in payload.items():
        if key in PERMITTED_FIELDS:
            updated_item[key] = value
    return updated_item
