PERMITTED_FIELDS = ["price", "description", "stock"]

def amend_product_features(item, payload):
    changes = {}
    for field, value in payload.items():
        if field in PERMITTED_FIELDS and field in item:
            item[field] = value
            changes[field] = value
    return changes
