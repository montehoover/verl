PERMITTED_FIELDS = ["price", "description", "stock"]

def update_item_information(product_info, new_data):
    updated_product = product_info.copy()
    
    for field, value in new_data.items():
        if field in PERMITTED_FIELDS:
            updated_product[field] = value
    
    return updated_product
