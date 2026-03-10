PERMITTED_FIELDS = ["price", "description", "stock"]

def validate_field(field, permitted_fields):
    return field in permitted_fields

def update_field(item, field, value):
    if field in item:
        item[field] = value
        return True
    return False

def amend_product_features(item, payload):
    changes = {}
    for field, value in payload.items():
        if validate_field(field, PERMITTED_FIELDS) and update_field(item, field, value):
            changes[field] = value
    return changes
