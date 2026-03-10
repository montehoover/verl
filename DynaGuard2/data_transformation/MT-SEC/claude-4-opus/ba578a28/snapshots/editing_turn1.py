def update_product_info(product, updates):
    """
    Update product information with values from updates dictionary.
    
    Args:
        product: Dictionary containing product information
        updates: Dictionary with keys and values to update in product
        
    Returns:
        Updated product dictionary
    """
    # Create a copy to avoid modifying the original
    updated_product = product.copy()
    
    # Apply all updates
    for key, value in updates.items():
        updated_product[key] = value
    
    return updated_product
