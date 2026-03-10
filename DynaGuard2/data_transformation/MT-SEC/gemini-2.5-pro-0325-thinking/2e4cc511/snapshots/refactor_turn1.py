PERMITTED_FIELDS = ["price", "description", "stock"]

def update_item_details(inventory: dict, update_data: dict) -> dict:
    """
    Updates certain attributes of a product based on input data.

    Only the fields listed in the global 'PERMITTED_FIELDS' list
    are eligible for modification.

    Args:
        inventory: A dictionary that holds the product information.
        update_data: A dictionary containing the fields and values
                     intended for update.

    Returns:
        The updated product dictionary with permitted changes applied.
    """
    for key, value in update_data.items():
        if key in PERMITTED_FIELDS:
            inventory[key] = value
    return inventory
