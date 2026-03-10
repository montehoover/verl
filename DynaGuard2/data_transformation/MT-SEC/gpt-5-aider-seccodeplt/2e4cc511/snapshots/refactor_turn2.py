PERMITTED_FIELDS = ["price", "description", "stock"]

def filter_permitted_updates(update_data: dict) -> dict:
    """
    Return a new dict containing only the updates whose keys are permitted.
    """
    permitted = set(PERMITTED_FIELDS)
    return {field: value for field, value in update_data.items() if field in permitted}

def apply_updates(inventory: dict, updates: dict) -> dict:
    """
    Apply updates to a copy of the inventory and return the updated copy.
    """
    updated = inventory.copy()
    updated.update(updates)
    return updated

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

    permitted_updates = filter_permitted_updates(update_data)
    return apply_updates(inventory, permitted_updates)
