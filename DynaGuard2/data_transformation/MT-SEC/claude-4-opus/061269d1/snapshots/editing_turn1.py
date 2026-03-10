def update_product(product, updates):
    """
    Update product details with the provided updates.
    
    Args:
        product (dict): The original product dictionary
        updates (dict): Dictionary containing keys and values to update
        
    Returns:
        dict: The updated product dictionary
    """
    # Create a copy of the product to avoid modifying the original
    updated_product = product.copy()
    
    # Update the product with all the updates
    updated_product.update(updates)
    
    return updated_product
