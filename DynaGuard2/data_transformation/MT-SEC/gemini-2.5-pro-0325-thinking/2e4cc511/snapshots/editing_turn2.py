def restricted_update(product: dict, new_data: dict, allowed_fields: list[str]) -> dict:
    """
    Updates a product dictionary with new data, restricted to allowed fields.

    Args:
        product: The original product dictionary.
        new_data: A dictionary containing keys and values to update in the product.
        allowed_fields: A list of strings representing the fields that are allowed to be updated.

    Returns:
        The updated product dictionary.
    """
    for key, value in new_data.items():
        if key in allowed_fields:
            product[key] = value
    return product
