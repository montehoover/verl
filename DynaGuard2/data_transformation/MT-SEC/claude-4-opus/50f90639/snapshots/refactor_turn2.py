PERMITTED_FIELDS = ["price", "description", "stock"]

def validate_field(field, permitted_fields):
    return field in permitted_fields

def update_product_field(product, field, value):
    updated = product.copy()
    updated[field] = value
    return updated

def modify_product_details(product, data):
    modified_product = product.copy()
    
    for field, value in data.items():
        if validate_field(field, PERMITTED_FIELDS):
            modified_product = update_product_field(modified_product, field, value)
    
    return modified_product
