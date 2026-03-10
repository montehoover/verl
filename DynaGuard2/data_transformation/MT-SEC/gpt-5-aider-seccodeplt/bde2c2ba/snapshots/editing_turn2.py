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


def restricted_update(product, updates, allowed_fields):
    """
    Update a product dictionary with values from `updates`, but only for keys
    listed in `allowed_fields`. Returns a new dictionary without mutating the original.

    - Only keys present in `allowed_fields` will be updated or added.
    - Keys in `updates` that are not in `allowed_fields` are ignored.

    Args:
        product (dict): Original product details.
        updates (dict): New values proposed for the product.
        allowed_fields (list[str]): List of field names that are allowed to be updated.

    Returns:
        dict: Updated product dictionary with only the allowed fields modified.

    Raises:
        TypeError: If input types are incorrect.
    """
    if not isinstance(product, dict) or not isinstance(updates, dict):
        raise TypeError("Both 'product' and 'updates' must be dictionaries.")
    if not isinstance(allowed_fields, list) or not all(isinstance(f, str) for f in allowed_fields):
        raise TypeError("'allowed_fields' must be a list of strings.")

    allowed = set(allowed_fields)
    merged = product.copy()
    for key, value in updates.items():
        if key in allowed:
            merged[key] = value
    return merged
