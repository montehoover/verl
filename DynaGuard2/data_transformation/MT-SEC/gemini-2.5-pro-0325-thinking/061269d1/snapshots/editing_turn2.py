def update_product(product: dict, updates: dict) -> dict:
    """
    Updates product details.

    Args:
        product: A dictionary representing the product.
        updates: A dictionary with keys and values to update in the product.

    Returns:
        The updated product dictionary.
    """
    product.update(updates)
    return product


def restricted_update(product: dict, updates: dict, modifiable_fields: list[str]) -> dict:
    """
    Updates product details, restricting updates to a predefined set of fields.

    Args:
        product: A dictionary representing the product.
        updates: A dictionary with keys and values to update in the product.
        modifiable_fields: A list of strings representing the fields that are allowed to be updated.

    Returns:
        The updated product dictionary.
    """
    for key, value in updates.items():
        if key in modifiable_fields:
            product[key] = value
    return product
