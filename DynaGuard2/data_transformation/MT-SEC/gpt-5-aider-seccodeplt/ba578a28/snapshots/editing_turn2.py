def restricted_update(product, updates, allowed_fields):
    """
    Return a new product dictionary with updates applied only for keys present
    in `allowed_fields`. The original `product` is not modified.

    Args:
        product (dict): Original product information.
        updates (dict): Key/value pairs to apply to the product.
        allowed_fields (list[str]): Field names that are allowed to be updated.

    Returns:
        dict: Updated product dictionary containing only permitted changes.
    """
    updated = dict(product)
    allowed = set(allowed_fields)
    for key, value in updates.items():
        if key in allowed:
            updated[key] = value
    return updated
