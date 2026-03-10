def restricted_update(product: dict, updates: dict, allowed_fields: list[str]) -> dict:
    """
    Updates product information for allowed fields only.

    Args:
        product: A dictionary representing the product.
        updates: A dictionary with keys and values to update in the product.
        allowed_fields: A list of strings representing the fields that are allowed to be updated.

    Returns:
        The updated product dictionary.
    """
    for key, value in updates.items():
        if key in allowed_fields:
            product[key] = value
    return product
