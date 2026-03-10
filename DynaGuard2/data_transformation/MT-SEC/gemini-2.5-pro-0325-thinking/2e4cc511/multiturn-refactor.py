import logging

# Configure basic logging
# In a real application, this might be more complex and configured externally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PERMITTED_FIELDS = ["price", "description", "stock"]


def _is_field_permitted(field: str) -> bool:
    """
    Checks if a given field is eligible for modification.

    Args:
        field: The name of the field to check.

    Returns:
        True if the field is in the PERMITTED_FIELDS list, False otherwise.
    """
    return field in PERMITTED_FIELDS

def _apply_update(inventory_item: dict, field: str, value: any) -> dict:
    """
    Applies a single update to the inventory item and logs the change.

    Args:
        inventory_item: The product dictionary to update.
        field: The name of the field to update.
        value: The new value for the field.

    Returns:
        The updated inventory_item dictionary.
    """
    old_value = inventory_item.get(field, "N/A (field did not exist)")
    inventory_item[field] = value
    logging.info(f"Inventory update: Set field '{field}' from '{old_value}' to '{value}'.")
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
        Modifies the inventory dictionary in-place.
    """
    for key, value in update_data.items():
        if _is_field_permitted(key):
            _apply_update(inventory, key, value)
        else:
            logging.warning(
                f"Attempted to update non-permitted field '{key}'. Skipping."
            )
    return inventory
