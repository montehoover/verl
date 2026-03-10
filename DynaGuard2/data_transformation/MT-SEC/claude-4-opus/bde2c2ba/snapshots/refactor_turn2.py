PERMITTED_FIELDS = ["price", "description", "stock"]

def filter_permitted_fields(new_data, permitted_fields):
    return {field: value for field, value in new_data.items() if field in permitted_fields}

def apply_updates(product_info, filtered_updates):
    updated_product = product_info.copy()
    updated_product.update(filtered_updates)
    return updated_product

def update_item_information(product_info, new_data):
    filtered_updates = filter_permitted_fields(new_data, PERMITTED_FIELDS)
    return apply_updates(product_info, filtered_updates)
