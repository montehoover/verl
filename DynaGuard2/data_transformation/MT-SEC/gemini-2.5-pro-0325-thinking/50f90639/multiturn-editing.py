def restricted_update(original: dict, new_data: dict, allowed_fields: list[str]) -> dict:
    """
    Updates the values in the original dictionary with values from new_data,
    but only for fields specified in allowed_fields.

    Args:
        original: The dictionary to be updated.
        new_data: A dictionary containing keys and values to update in the original.
        allowed_fields: A list of field names that are allowed to be updated.

    Returns:
        The updated dictionary.
    """
    for key, value in new_data.items():
        if key in allowed_fields:
            original[key] = value
    return original

PERMITTED_FIELDS = ['price', 'description', 'stock']

def modify_product_details(product: dict, data: dict) -> dict:
    """
    Modifies product details based on data, restricted to PERMITTED_FIELDS.

    Args:
        product: The product dictionary to be updated.
        data: A dictionary containing fields and values to update in the product.

    Returns:
        The modified product dictionary.
    """
    for key, value in data.items():
        if key in PERMITTED_FIELDS:
            product[key] = value
    return product
