PERMITTED_FIELDS = ["price", "description", "stock"]

def is_field_permitted(field):
    return field in PERMITTED_FIELDS

def update_permitted_fields(item, details):
    updated_item = item.copy()
    for field, value in details.items():
        if is_field_permitted(field):
            updated_item[field] = value
    return updated_item

def edit_product_info(item, details):
    return update_permitted_fields(item, details)
