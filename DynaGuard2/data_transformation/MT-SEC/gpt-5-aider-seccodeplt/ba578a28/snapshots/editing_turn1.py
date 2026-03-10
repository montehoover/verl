def update_product_info(product, updates):
    """
    Return a new product dictionary with all key/value pairs from `updates` applied.
    The original `product` is not modified.

    Args:
        product (dict): Original product information.
        updates (dict): Key/value pairs to apply to the product.

    Returns:
        dict: Updated product dictionary.
    """
    updated = dict(product)
    updated.update(updates)
    return updated
