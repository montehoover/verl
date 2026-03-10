def update_product_info(product, updates):
    """Update product dictionary with values from updates dictionary."""
    updated_product = product.copy()
    updated_product.update(updates)
    return updated_product
