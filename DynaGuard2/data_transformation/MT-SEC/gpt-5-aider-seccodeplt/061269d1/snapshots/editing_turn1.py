from typing import Any, Dict

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
