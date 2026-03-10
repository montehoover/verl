def display_product_details(product_details):
    """
    Display product details in a formatted string.
    
    Args:
        product_details (dict): Dictionary containing product attributes
            - price: Product price
            - description: Product description
            - stock: Available stock quantity
            - category: Product category
    
    Returns:
        str: Formatted string with product details
    """
    price = product_details.get('price', 'N/A')
    description = product_details.get('description', 'No description available')
    stock = product_details.get('stock', 0)
    category = product_details.get('category', 'Uncategorized')
    
    formatted_details = f"""Product Details:
-----------------
Category: {category}
Price: ${price}
Stock: {stock} units
Description: {description}"""
    
    return formatted_details
