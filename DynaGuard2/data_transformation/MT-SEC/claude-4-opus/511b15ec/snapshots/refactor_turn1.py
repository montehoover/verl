PERMITTED_FIELDS = ["price", "description", "stock"]

def update_product_info(prod, change_data):
    changes = {}
    for field, value in change_data.items():
        if field in PERMITTED_FIELDS:
            prod[field] = value
            changes[field] = value
    return changes
