PERMITTED_FIELDS = ["price", "description", "stock"]


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


def modify_product_and_log(prod, change_data):
    """
    Update a product dictionary with changes and log each modification.
    
    Args:
        prod (dict): Product dictionary to be modified
        change_data (dict): Dictionary containing the changes to apply
        
    Returns:
        dict: Updated product dictionary
    """
    for key, new_value in change_data.items():
        if key in prod:
            old_value = prod[key]
            prod[key] = new_value
            print(f"Modified {key}: {old_value} -> {new_value}")
        else:
            prod[key] = new_value
            print(f"Added {key}: {new_value}")
    
    return prod


def update_product_info(prod, change_data):
    """
    Update a product dictionary with changes, only allowing updates to permitted fields.
    
    Args:
        prod (dict): Product dictionary to be updated
        change_data (dict): Dictionary containing the changes to apply
        
    Returns:
        dict: Updated product dictionary with only permitted changes applied
    """
    for key, value in change_data.items():
        if key in PERMITTED_FIELDS:
            prod[key] = value
    
    return prod
