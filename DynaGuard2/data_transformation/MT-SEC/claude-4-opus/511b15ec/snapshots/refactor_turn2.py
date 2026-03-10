PERMITTED_FIELDS = ["price", "description", "stock"]

def validate_fields(change_data, permitted_fields):
    """Validate and filter fields based on permitted fields list."""
    return {field: value for field, value in change_data.items() if field in permitted_fields}

def apply_updates(prod, validated_changes):
    """Apply validated changes to the product dictionary."""
    for field, value in validated_changes.items():
        prod[field] = value
    return validated_changes

def update_product_info(prod, change_data):
    validated_changes = validate_fields(change_data, PERMITTED_FIELDS)
    changes = apply_updates(prod, validated_changes)
    return changes
