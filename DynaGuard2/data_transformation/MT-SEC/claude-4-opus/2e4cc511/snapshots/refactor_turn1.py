PERMITTED_FIELDS = ["price", "description", "stock"]

def update_item_details(inventory, update_data):
    for field, value in update_data.items():
        if field in PERMITTED_FIELDS:
            inventory[field] = value
    return inventory
