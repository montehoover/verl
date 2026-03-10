PERMITTED_FIELDS = ['price', 'description', 'stock']

def update_item_details(inventory: dict, update_data: dict) -> dict:
    """
    Updates product details in the inventory, restricted to permitted fields.

    Args:
        inventory: The product information dictionary.
        update_data: A dictionary containing fields and values for update.

    Returns:
        The updated product dictionary.
    """
    for key, value in update_data.items():
        if key in PERMITTED_FIELDS:
            inventory[key] = value
    return inventory
