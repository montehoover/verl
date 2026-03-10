def update_product_info(product, updates):
    """Update product dictionary with values from updates dictionary."""
    updated_product = product.copy()
    updated_product.update(updates)
    return updated_product

def restricted_update(product, updates, allowed_fields):
    """Update product dictionary with values from updates dictionary, but only for allowed fields."""
    updated_product = product.copy()
    for field, value in updates.items():
        if field in allowed_fields:
            updated_product[field] = value
    return updated_product
