from typing import Any, Dict

# Only these fields are allowed to be updated
PERMITTED_FIELDS = ["price", "description", "stock"]

def update_item_details(inventory: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update permitted attributes of a product based on input data.

    Args:
        inventory: dict holding the product information (e.g., price, description, stock, category).
        update_data: dict containing fields and values intended for update.

    Returns:
        Dict[str, Any]: The updated product dictionary with permitted changes applied.
    """
    if not isinstance(inventory, dict):
        raise TypeError("inventory must be a dict")
    if not isinstance(update_data, dict):
        raise TypeError("update_data must be a dict")

    # Work on a copy to avoid mutating the original input
    updated_inventory = inventory.copy()

    for field, value in update_data.items():
        if field in PERMITTED_FIELDS:
            updated_inventory[field] = value

    return updated_inventory
