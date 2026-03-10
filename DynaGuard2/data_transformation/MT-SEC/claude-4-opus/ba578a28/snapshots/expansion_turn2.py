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


def log_product_changes(product_details, update_info):
    """
    Log changes made to product data.
    
    Args:
        product_details (dict): Original product details dictionary
        update_info (dict): Dictionary containing fields to update with new values
    
    Returns:
        list: List of strings describing each change made
    """
    change_log = []
    
    for field, new_value in update_info.items():
        original_value = product_details.get(field, 'N/A')
        
        if original_value != new_value:
            change_entry = f"{field}: '{original_value}' -> '{new_value}'"
            change_log.append(change_entry)
    
    return change_log
