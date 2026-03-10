from typing import Any, Dict
from copy import deepcopy

PERMITTED_FIELDS = ["price", "description", "stock"]


def update_product_info(prod: Dict[str, Any], change_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update only fields listed in the global PERMITTED_FIELDS.
    Disregard any other fields. Return a dictionary containing the modified product.

    Expects a global:
      PERMITTED_FIELDS = ["price", "description", "stock"]
    """
    if not isinstance(prod, dict):
        raise TypeError("prod must be a dict")
    if not isinstance(change_data, dict):
        raise TypeError("change_data must be a dict")

    updated = deepcopy(prod)

    for field, value in change_data.items():
        if field in PERMITTED_FIELDS:
            updated[field] = value

    return {"product": updated}
