from typing import Any, Dict, List

PERMITTED_FIELDS = ['price', 'description', 'stock']

__all__ = ["selective_update", "amend_product_features"]


def selective_update(product: Dict[str, Any], new_data: Dict[str, Any], allowed_fields: List[str]) -> Dict[str, Any]:
    """
    Return a new dictionary with values from `product` selectively updated by `new_data`,
    but only for keys listed in `allowed_fields`.

    This performs a shallow merge:
    - Only keys present in `allowed_fields` will be taken from `new_data`.
    - Keys in `allowed_fields` that are not in `new_data` are ignored.
    - The original `product` dictionary is not mutated.

    :param product: Original product dictionary.
    :param new_data: Dictionary containing keys/values to update.
    :param allowed_fields: List of field names that are allowed to be updated.
    :return: A new merged dictionary with allowed updates applied.
    """
    if not isinstance(product, dict):
        raise TypeError("product must be a dict")
    if not isinstance(new_data, dict):
        raise TypeError("new_data must be a dict")
    if not isinstance(allowed_fields, list):
        raise TypeError("allowed_fields must be a list of strings")
    if not all(isinstance(field, str) for field in allowed_fields):
        raise TypeError("allowed_fields must be a list of strings")

    allowed = set(allowed_fields)

    updated = product.copy()
    for key, value in new_data.items():
        if key in allowed:
            updated[key] = value
    return updated


def amend_product_features(item: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a new dictionary with `item` updated using `payload`, but only for fields
    listed in PERMITTED_FIELDS. Fields not in PERMITTED_FIELDS are ignored.

    :param item: The original product dictionary (e.g., includes 'price', 'description', 'stock', 'category').
    :param payload: Dictionary with new values for fields to update.
    :return: A new product dictionary with permitted changes applied.
    """
    return selective_update(item, payload, PERMITTED_FIELDS)
