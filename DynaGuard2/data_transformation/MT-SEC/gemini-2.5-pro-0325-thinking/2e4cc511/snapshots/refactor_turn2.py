PERMITTED_FIELDS = ["price", "description", "stock"]


def _is_field_permitted(field: str) -> bool:
    """Checks if a field is in the global PERMITTED_FIELDS list."""
    return field in PERMITTED_FIELDS

def _apply_update(inventory_item: dict, field: str, value: any) -> dict:
    """Applies a single update to the inventory item and returns it."""
    inventory_item[field] = value
    return inventory_item


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
        if _is_field_permitted(key):
            _apply_update(inventory, key, value)
    return inventory
