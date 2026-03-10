PERMITTED_FIELDS = ["price", "description", "stock"]

def modify_product_details(product, data):
    modified_product = product.copy()
    
    for field, value in data.items():
        if field in PERMITTED_FIELDS:
            modified_product[field] = value
    
    return modified_product
