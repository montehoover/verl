PERMITTED_FIELDS = ["price", "description", "stock"]

def _is_field_permitted(field_name: str, allowed_fields: list[str]) -> bool:
    """
    Checks if a field name is present in the list of allowed fields.

    Args:
        field_name: The name of the field to check.
        allowed_fields: A list of field names that are allowed.

    Returns:
        True if the field is permitted, False otherwise.
    """
    return field_name in allowed_fields

def _apply_update_to_data(data: dict, field_name: str, new_value: any) -> dict:
    """
    Applies an update for a single field to a copy of the data.
    Returns a new dictionary with the field updated. This is a pure function.

    Args:
        data: The dictionary to base the update on.
        field_name: The name of the field to update.
        new_value: The new value for the field.

    Returns:
        A new dictionary with the specified field updated.
    """
    updated_data = data.copy()
    updated_data[field_name] = new_value
    return updated_data

def edit_product_info(item: dict, details: dict) -> dict:
    """
    Updates certain attributes of a product based on input data.

    Only the fields listed in the global 'PERMITTED_FIELDS' list
    are eligible for modification.

    Args:
        item: dict, a dictionary that holds the product information.
              This dictionary will be modified in-place.
        details: dict, a dictionary containing the fields and values
                 intended for update.

    Returns:
        The updated product dictionary with permitted changes applied (same object as input 'item').
    """
    working_item_state = item.copy()  # Start with a shallow copy for internal processing

    for key, value in details.items():
        if _is_field_permitted(key, PERMITTED_FIELDS):
            working_item_state = _apply_update_to_data(working_item_state, key, value)

    # Apply the accumulated changes from working_item_state back to the original item
    item.clear()
    item.update(working_item_state)
    
    return item
