def update_product_info(product, new_data):
    """
    Update product dictionary with new data.
    
    Args:
        product: Dictionary containing product information
        new_data: Dictionary with keys and values to update in product
        
    Returns:
        Updated product dictionary
    """
    updated_product = product.copy()
    updated_product.update(new_data)
    return updated_product
