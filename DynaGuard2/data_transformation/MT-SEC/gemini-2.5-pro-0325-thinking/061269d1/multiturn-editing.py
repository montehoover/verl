PERMITTED_FIELDS = ['price', 'description', 'stock']

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


def edit_product_info(item: dict, details: dict) -> dict:
    """
    Updates product information, allowing only permitted fields to be modified.

    Args:
        item: A dictionary representing the product details.
        details: A dictionary of fields and values for update.

    Returns:
        The updated product dictionary.
    """
    for key, value in details.items():
        if key in PERMITTED_FIELDS:
            item[key] = value
    return item
