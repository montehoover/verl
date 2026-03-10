from typing import Any, Dict

__all__ = ["update_product_info"]


def update_product_info(product: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a new dictionary with values from `product` updated by `new_data`.

    This performs a shallow merge:
    - Keys in `new_data` will overwrite keys in `product`.
    - The original `product` dictionary is not mutated.

    :param product: Original product dictionary.
    :param new_data: Dictionary containing keys/values to update.
    :return: A new merged dictionary.
    """
    if not isinstance(product, dict):
        raise TypeError("product must be a dict")
    if not isinstance(new_data, dict):
        raise TypeError("new_data must be a dict")

    updated = product.copy()
    updated.update(new_data)
    return updated
