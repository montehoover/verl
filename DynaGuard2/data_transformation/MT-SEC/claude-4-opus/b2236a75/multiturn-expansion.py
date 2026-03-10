# Sample orders dictionary for testing
orders = {
    "ORD001": {"id": "ORD001", "customer": "John Doe", "total": 99.99, "status": "shipped"},
    "ORD002": {"id": "ORD002", "customer": "Jane Smith", "total": 149.50, "status": "processing"},
    "ORD003": {"id": "ORD003", "customer": "Bob Johnson", "total": 75.00, "status": "delivered"}
}

def get_order_by_id(order_id):
    """
    Retrieve an order by its ID.
    
    Args:
        order_id (str): The ID of the order to retrieve
        
    Returns:
        dict: The order object containing order details
        
    Raises:
        ValueError: If the order ID doesn't exist
    """
    if order_id not in orders:
        raise ValueError(f"Order with ID '{order_id}' not found")
    
    return orders[order_id]


def check_template_placeholders(template):
    """
    Verify that all placeholders in a template string are valid and safe.
    
    Args:
        template (str): The template string containing placeholders like {customer.name}
        
    Returns:
        bool: True if all placeholders are valid
        
    Raises:
        ValueError: If any placeholder is invalid or attempts to access sensitive information
    """
    import re
    
    # Define allowed placeholder patterns
    allowed_placeholders = {
        'customer.name',
        'customer.email',
        'order.id',
        'order.total',
        'order.status',
        'order.date',
        'product.name',
        'product.price',
        'product.quantity'
    }
    
    # Define sensitive patterns that should be blocked
    sensitive_patterns = {
        'password',
        'credit_card',
        'ssn',
        'api_key',
        'secret',
        'token',
        'private'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    for placeholder in placeholders:
        # Check for sensitive information
        placeholder_lower = placeholder.lower()
        for sensitive in sensitive_patterns:
            if sensitive in placeholder_lower:
                raise ValueError(f"Invalid placeholder '{{{placeholder}}}': attempts to access sensitive information")
        
        # Check if placeholder is in allowed list
        if placeholder not in allowed_placeholders:
            raise ValueError(f"Invalid placeholder '{{{placeholder}}}': not in allowed list")
    
    return True


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


def generate_customer_receipt(order_id, template):
    """
    Generate a customer receipt by replacing placeholders with actual order data.
    
    Args:
        order_id (str): The ID of the order to generate a receipt for
        template (str): The template string containing placeholders
        
    Returns:
        str: The generated receipt with placeholders replaced
        
    Raises:
        ValueError: If order not found, template invalid, or sensitive info accessed
    """
    import re
    
    # Validate template first
    try:
        check_template_placeholders(template)
    except ValueError as e:
        raise e
    
    # Get the order
    if order_id not in orders:
        raise ValueError(f"Order with ID '{order_id}' not found")
    
    order = orders[order_id]
    
    # Create replacement dictionary
    replacements = {
        'customer.name': order.customer.name,
        'customer.email': order.customer.email,
        'order.id': order.id,
        'order.total': str(order.total),
        'order.status': getattr(order, 'status', 'pending'),
        'order.date': getattr(order, 'date', 'N/A'),
        'product.name': 'N/A',
        'product.price': 'N/A',
        'product.quantity': 'N/A'
    }
    
    # Replace placeholders in template
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(f'{{{placeholder}}}', value)
    
    return result
