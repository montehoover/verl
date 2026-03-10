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


class Customer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card


class Order:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method


orders = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item 1", "Item 2"],
                    99.99,
                    "Credit Card")
}


def create_order_receipt(purchase_id, receipt_template):
    """
    Generate a complete order receipt using an order ID and template.
    
    Args:
        purchase_id (str): The order ID to generate receipt for
        receipt_template (str): Template string with placeholders
        
    Returns:
        str: The formatted receipt string
        
    Raises:
        ValueError: If order ID is nonexistent or a restricted placeholder is used
    """
    import re
    
    # Check if order exists
    if purchase_id not in orders:
        raise ValueError(f"Order ID '{purchase_id}' not found")
    
    order = orders[purchase_id]
    
    # Define restricted placeholders
    restricted_placeholders = ['customer.credit_card', 'customer.address']
    
    # Check for restricted placeholders in template
    placeholders = re.findall(r'\{([^}]+)\}', receipt_template)
    for placeholder in placeholders:
        if placeholder in restricted_placeholders:
            raise ValueError(f"Restricted placeholder used: {{{placeholder}}}")
    
    # Create values dictionary for replacement
    values = {
        'order': {
            'id': order.id,
            'items': ', '.join(order.items),
            'total': order.total,
            'payment_method': order.payment_method
        },
        'customer': {
            'name': order.customer.name,
            'email': order.customer.email,
            'address': order.customer.address,
            'credit_card': order.customer.credit_card
        }
    }
    
    # Use replace_placeholders to format the receipt
    return replace_placeholders(receipt_template, values)
