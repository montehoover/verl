from typing import Any, Dict, List

def update_product(product: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a product dictionary with values from the updates dictionary.

    This function modifies the product in place and returns it.

    Args:
        product: The original product dictionary to update.
        updates: A dictionary of keys and values to set on the product.

    Returns:
        The updated product dictionary.
    """
    if not isinstance(product, dict):
        raise TypeError("product must be a dictionary")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dictionary")

    product.update(updates)
    return product


def restricted_update(
    product: Dict[str, Any],
    updates: Dict[str, Any],
    modifiable_fields: List[str],
) -> Dict[str, Any]:
    """
    Update only allowed fields on a product dictionary.

    This function modifies the product in place and returns it.

    Args:
        product: The original product dictionary to update.
        updates: A dictionary of keys and values to set on the product.
        modifiable_fields: A list of field names that are allowed to be updated.

    Returns:
        The updated product dictionary, with only allowed fields modified.
    """
    if not isinstance(product, dict):
        raise TypeError("product must be a dictionary")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dictionary")
    if not isinstance(modifiable_fields, list):
        raise TypeError("modifiable_fields must be a list of strings")
    if not all(isinstance(field, str) for field in modifiable_fields):
        raise TypeError("all items in modifiable_fields must be strings")

    allowed = set(modifiable_fields)
    for key, value in updates.items():
        if key in allowed:
            product[key] = value

    return product
