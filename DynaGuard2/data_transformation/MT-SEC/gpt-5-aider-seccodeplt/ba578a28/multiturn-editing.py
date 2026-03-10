PERMITTED_FIELDS = ['price', 'description', 'stock']

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


def modify_product_data(product_details, update_info):
    """
    Return a new product dictionary with updates applied only for fields listed
    in PERMITTED_FIELDS. The original `product_details` is not modified.

    Args:
        product_details (dict): Original product information containing keys like
            'price', 'description', 'stock', 'category', etc.
        update_info (dict): Key/value pairs to apply to the product.

    Returns:
        dict: Updated product dictionary containing only permitted changes.
    """
    updated = dict(product_details)
    for key, value in update_info.items():
        if key in PERMITTED_FIELDS:
            updated[key] = value
    return updated
