def get_product_info(prod):
    """
    Retrieve product details from a product dictionary and return a formatted string.
    
    Args:
        prod (dict): Product dictionary containing price, description, stock, and category
        
    Returns:
        str: Formatted string listing product attributes and their values
    """
    attributes = ['price', 'description', 'stock', 'category']
    result = []
    
    for attr in attributes:
        if attr in prod:
            result.append(f"{attr}: {prod[attr]}")
    
    return '\n'.join(result)
