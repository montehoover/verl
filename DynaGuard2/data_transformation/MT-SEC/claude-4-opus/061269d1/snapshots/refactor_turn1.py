PERMITTED_FIELDS = ["price", "description", "stock"]

def edit_product_info(item, details):
    for field, value in details.items():
        if field in PERMITTED_FIELDS:
            item[field] = value
    return item
