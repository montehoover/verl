def simple_update(product, new_data):
    """
    Update product details by merging new_data into product dictionary.
    
    Args:
        product: Dictionary containing product details
        new_data: Dictionary with keys and values to update in product
    
    Returns:
        Updated product dictionary
    """
    updated_product = product.copy()
    updated_product.update(new_data)
    return updated_product
