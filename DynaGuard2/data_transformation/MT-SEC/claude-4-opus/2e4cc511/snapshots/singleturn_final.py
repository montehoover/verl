PERMITTED_FIELDS = ["price", "description", "stock"]

def update_item_details(inventory, update_data):
    """
    Updates certain attributes of a product based on input data.
    Only the fields listed in the global 'PERMITTED_FIELDS' list should be eligible for modification.
    
    Args:
        inventory (dict): A dictionary that holds the product information.
        update_data (dict): A dictionary containing the fields and values intended for update.
    
    Returns:
        dict: The updated product dictionary with permitted changes applied.
    """
    # Create a copy of the inventory to avoid modifying the original
    updated_inventory = inventory.copy()
    
    # Iterate through the update_data and apply only permitted changes
    for field, value in update_data.items():
        if field in PERMITTED_FIELDS:
            updated_inventory[field] = value
    
    return updated_inventory
