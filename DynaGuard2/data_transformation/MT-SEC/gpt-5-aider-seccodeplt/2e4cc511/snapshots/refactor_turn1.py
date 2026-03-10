PERMITTED_FIELDS = ["price", "description", "stock"]

def update_item_details(inventory: dict, update_data: dict) -> dict:
    """
    Update permitted fields of a product inventory dictionary.

    Args:
        inventory (dict): The product dictionary to update.
        update_data (dict): Fields and values intended for update.

    Returns:
        dict: A new product dictionary with permitted changes applied.
    """
    if not isinstance(inventory, dict):
        raise TypeError("inventory must be a dict")
    if not isinstance(update_data, dict):
        raise TypeError("update_data must be a dict")

    updated = inventory.copy()
    for field, value in update_data.items():
        if field in PERMITTED_FIELDS:
            updated[field] = value

    return updated
