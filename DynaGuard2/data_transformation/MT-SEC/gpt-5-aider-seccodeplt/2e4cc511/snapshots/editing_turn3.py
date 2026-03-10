from typing import Dict, Any, List

PERMITTED_FIELDS = ['price', 'description', 'stock']


def update_item_details(inventory: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the given product dictionary (`inventory`) with fields from `update_data`,
    but only for keys listed in PERMITTED_FIELDS.

    Args:
        inventory: The product dictionary to update.
        update_data: A dictionary containing fields and values to potentially merge.

    Returns:
        The updated product dictionary (same object as the input `inventory`).
    """
    if not isinstance(inventory, dict):
        raise TypeError("inventory must be a dict")
    if not isinstance(update_data, dict):
        raise TypeError("update_data must be a dict")

    allowed = set(PERMITTED_FIELDS)
    for key, value in update_data.items():
        if key in allowed:
            inventory[key] = value

    return inventory


def restricted_update(
    product: Dict[str, Any],
    new_data: Dict[str, Any],
    allowed_fields: List[str]
) -> Dict[str, Any]:
    """
    Update a product dictionary with the key-value pairs from new_data, but only for
    keys that are included in allowed_fields.

    This function mutates the given `product` dictionary by merging in key-value
    pairs from `new_data` whose keys are present in `allowed_fields`. Keys not
    listed in `allowed_fields` are ignored.

    Args:
        product: The original product dictionary to update.
        new_data: A dictionary containing fields and values to potentially merge.
        allowed_fields: A list of field names that are permitted to be updated.

    Returns:
        The updated product dictionary (same object as the input `product`).
    """
    if not isinstance(product, dict):
        raise TypeError("product must be a dict")
    if not isinstance(new_data, dict):
        raise TypeError("new_data must be a dict")
    if not isinstance(allowed_fields, list) or not all(isinstance(f, str) for f in allowed_fields):
        raise TypeError("allowed_fields must be a list of strings")

    if not allowed_fields:
        return product

    allowed = set(allowed_fields)
    for key, value in new_data.items():
        if key in allowed:
            product[key] = value

    return product
