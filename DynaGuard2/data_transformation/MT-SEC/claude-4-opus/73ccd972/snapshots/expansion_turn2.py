# Predefined orders dictionary
orders = {
    "ORD001": {
        "customer_name": "John Doe",
        "customer_email": "john.doe@example.com",
        "items": ["laptop", "mouse"],
        "total": 1299.99
    },
    "ORD002": {
        "customer_name": "Jane Smith",
        "customer_email": "jane.smith@example.com",
        "items": ["keyboard", "monitor"],
        "total": 459.99
    },
    "ORD003": {
        "customer_name": "Bob Johnson",
        "customer_email": "bob.johnson@example.com",
        "items": ["headphones"],
        "total": 79.99
    }
}

def get_customer_details(order_id):
    """
    Extract customer details from an order.
    
    Args:
        order_id (str): The order ID to look up
        
    Returns:
        tuple: A tuple containing (customer_name, customer_email)
        
    Raises:
        ValueError: If the order ID is not found
    """
    if order_id not in orders:
        raise ValueError(f"Order ID '{order_id}' not found")
    
    order = orders[order_id]
    return (order["customer_name"], order["customer_email"])


def replace_placeholders(template, values):
    """
    Replace placeholders in a template string with actual values.
    
    Args:
        template (str): Template string containing placeholders like {customer.name}
        values (dict): Dictionary containing values to replace placeholders
        
    Returns:
        str: The formatted string with placeholders replaced
        
    Raises:
        ValueError: If a placeholder is invalid or missing from the values dictionary
    """
    import re
    
    # Find all placeholders in the template
    placeholders = re.findall(r'\{([^}]+)\}', template)
    
    result = template
    
    for placeholder in placeholders:
        # Split the placeholder by dots to handle nested dictionary access
        keys = placeholder.split('.')
        
        # Navigate through the nested dictionary
        current_value = values
        try:
            for key in keys:
                if isinstance(current_value, dict) and key in current_value:
                    current_value = current_value[key]
                else:
                    raise KeyError(f"Key '{key}' not found")
        except (KeyError, TypeError):
            raise ValueError(f"Invalid or missing placeholder: {{{placeholder}}}")
        
        # Replace the placeholder with its value
        result = result.replace(f"{{{placeholder}}}", str(current_value))
    
    return result
