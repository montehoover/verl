PERMITTED_FIELDS = ["price", "description", "stock"]

def is_field_permitted(field):
    """Check if a field is in the permitted fields list."""
    return field in PERMITTED_FIELDS

def apply_field_update(product_details, field, value):
    """Apply a single field update to the product."""
    product_details[field] = value
    return product_details

def filter_permitted_updates(update_info):
    """Filter update_info to only include permitted fields."""
    return {field: value for field, value in update_info.items() if is_field_permitted(field)}

def modify_product_data(product_details, update_info):
    permitted_updates = filter_permitted_updates(update_info)
    for field, value in permitted_updates.items():
        apply_field_update(product_details, field, value)
    return product_details
