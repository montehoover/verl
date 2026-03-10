PERMITTED_FIELDS = ["price", "description", "stock"]

def filter_permitted_fields(update_data, permitted_fields):
    return {field: value for field, value in update_data.items() if field in permitted_fields}

def apply_updates(inventory, filtered_updates):
    updated_inventory = inventory.copy()
    updated_inventory.update(filtered_updates)
    return updated_inventory

def update_item_details(inventory, update_data):
    filtered_updates = filter_permitted_fields(update_data, PERMITTED_FIELDS)
    return apply_updates(inventory, filtered_updates)
