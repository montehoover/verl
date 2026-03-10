from typing import Any, Dict

# Allowed fields that can be updated
PERMITTED_FIELDS = ["price", "description", "stock"]


def amend_product_features(item: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the given product item using values from payload, but only for fields
    listed in the global PERMITTED_FIELDS.

    Args:
        item: dict representing the product (e.g., {"price": 100, "description": "...", "stock": 50, "category": "tools"})
        payload: dict containing new values for fields that need to be updated.

    Returns:
        dict: The updated product dictionary (item) reflecting the applied changes.

    Notes:
        - Only keys present in PERMITTED_FIELDS are updated.
        - Keys in payload not in PERMITTED_FIELDS are ignored.
    """
    if not isinstance(item, dict):
        raise TypeError("item must be a dict")
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    for key, value in payload.items():
        if key in PERMITTED_FIELDS:
            item[key] = value

    return item
