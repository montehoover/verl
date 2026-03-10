def update_product_info(product, updates):
    """
    Merge two dictionaries representing a product and its updates.

    - Returns a new dictionary with keys from `product` overridden or added
      by values from `updates`.
    - Does not mutate the original `product` dictionary.

    Args:
        product (dict): Original product details.
        updates (dict): New values to merge into the product.

    Returns:
        dict: Updated product dictionary.
    """
    if not isinstance(product, dict) or not isinstance(updates, dict):
        raise TypeError("Both 'product' and 'updates' must be dictionaries.")

    merged = product.copy()
    merged.update(updates)
    return merged
