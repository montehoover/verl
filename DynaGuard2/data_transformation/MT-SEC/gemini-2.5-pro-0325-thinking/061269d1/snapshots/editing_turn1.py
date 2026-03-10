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
