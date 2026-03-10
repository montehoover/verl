PERMITTED_FIELDS = ["price", "description", "stock"]

def modify_product_data(product_details, update_info):
    for field, value in update_info.items():
        if field in PERMITTED_FIELDS:
            product_details[field] = value
    return product_details
