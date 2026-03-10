PERMITTED_FIELDS = ["price", "description", "stock"]


def validate_field(field, permitted_fields):
    """
    Check if a field is allowed to be modified.
    
    Args:
        field (str): The name of the field to validate.
        permitted_fields (list): List of field names that are allowed to be modified.
    
    Returns:
        bool: True if the field is in the permitted fields list, False otherwise.
    """
    return field in permitted_fields


def update_field(item, field, value):
    """
    Update a specific field in the item dictionary if it exists.
    
    Args:
        item (dict): The product dictionary to update.
        field (str): The name of the field to update.
        value: The new value to assign to the field.
    
    Returns:
        bool: True if the field was successfully updated, False if the field
              doesn't exist in the item.
    """
    if field in item:
        item[field] = value
        return True
    return False


def amend_product_features(item, payload):
    """
    Update product fields with new values from the payload.
    
    Only fields present in the PERMITTED_FIELDS list can be modified.
    The function will only update fields that exist in the original item.
    
    Args:
        item (dict): The dictionary representing the product object with its
                     corresponding fields.
        payload (dict): A dictionary containing the new values for the fields
                        that need to be updated.
    
    Returns:
        dict: A dictionary reflecting the changes made to the product object,
              containing only the fields that were successfully updated.
    """
    changes = {}
    
    # Iterate through each field-value pair in the payload
    for field, value in payload.items():
        # Check if the field is permitted and successfully updated
        if validate_field(field, PERMITTED_FIELDS) and update_field(item, field, value):
            # Track the successful change
            changes[field] = value
    
    return changes
