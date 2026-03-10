from typing import Dict, Any


def simple_update(product: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a product dictionary with the key-value pairs from new_data.

    This function mutates the given `product` dictionary by merging in all key-value
    pairs from `new_data` without any restrictions. If a key already exists in
    `product`, its value will be overwritten by the value from `new_data`.

    Args:
        product: The original product dictionary to update.
        new_data: A dictionary containing fields and values to merge into the product.

    Returns:
        The updated product dictionary (same object as the input `product`).
    """
    if not isinstance(product, dict):
        raise TypeError("product must be a dict")
    if not isinstance(new_data, dict):
        raise TypeError("new_data must be a dict")

    product.update(new_data)
    return product
