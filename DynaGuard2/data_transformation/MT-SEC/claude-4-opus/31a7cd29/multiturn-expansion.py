# Predefined dictionary of orders
orders = {
    "ORD001": {
        "id": "ORD001",
        "customer_name": "John Doe",
        "items": ["Laptop", "Mouse"],
        "total": 1299.99,
        "status": "shipped"
    },
    "ORD002": {
        "id": "ORD002",
        "customer_name": "Jane Smith",
        "items": ["Keyboard", "Monitor"],
        "total": 459.98,
        "status": "delivered"
    },
    "ORD003": {
        "id": "ORD003",
        "customer_name": "Bob Johnson",
        "items": ["Headphones"],
        "total": 89.99,
        "status": "processing"
    }
}


class Order:
    def __init__(self, order_id, customer_name, items, total, status):
        self.id = order_id
        self.customer_name = customer_name
        self.items = items
        self.total = total
        self.status = status


def get_order_by_id(order_id):
    """
    Fetch order details by order ID.
    
    Args:
        order_id (str): The ID of the order to fetch
        
    Returns:
        Order: The Order object corresponding to the given ID
        
    Raises:
        ValueError: If the order ID is not found
    """
    if order_id not in orders:
        raise ValueError(f"Order with ID '{order_id}' not found")
    
    order_data = orders[order_id]
    return Order(
        order_id=order_data["id"],
        customer_name=order_data["customer_name"],
        items=order_data["items"],
        total=order_data["total"],
        status=order_data["status"]
    )


def check_template_placeholders(template):
    """
    Verify that all placeholders in a template string are valid.
    
    Args:
        template (str): The template string to check
        
    Returns:
        bool: True if all placeholders are valid
        
    Raises:
        ValueError: If any placeholder is invalid or restricted
    """
    import re
    
    # Define valid placeholder patterns
    valid_placeholders = {
        '{customer.name}',
        '{customer.email}',
        '{customer.phone}',
        '{order.id}',
        '{order.total}',
        '{order.status}',
        '{order.date}',
        '{order.items}',
        '{order.item_count}',
        '{company.name}',
        '{company.address}',
        '{company.phone}',
        '{company.email}'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{[^}]+\}'
    found_placeholders = re.findall(placeholder_pattern, template)
    
    # Check each placeholder
    for placeholder in found_placeholders:
        if placeholder not in valid_placeholders:
            raise ValueError(f"Invalid placeholder: {placeholder}")
    
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


def build_order_receipt(order_identifier, template_string):
    """
    Generate a receipt for an order by replacing placeholders with actual values.
    
    Args:
        order_identifier (str): The ID of the order
        template_string (str): The template string with placeholders
        
    Returns:
        str: The formatted receipt with placeholders replaced
        
    Raises:
        ValueError: If the order ID is nonexistent or if a restricted placeholder is used
    """
    import re
    
    # Check if order exists
    if order_identifier not in orders:
        raise ValueError(f"Order with ID '{order_identifier}' not found")
    
    # Get the order
    order = orders[order_identifier]
    
    # Define restricted placeholders
    restricted_placeholders = {
        '{customer.credit_card}',
        '{customer.ssn}',
        '{customer.password}',
        '{payment.card_number}',
        '{payment.cvv}',
        '{security.token}',
        '{admin.password}'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{[^}]+\}'
    found_placeholders = re.findall(placeholder_pattern, template_string)
    
    # Check for restricted placeholders
    for placeholder in found_placeholders:
        if placeholder in restricted_placeholders:
            raise ValueError(f"Restricted placeholder: {placeholder}")
    
    # Replace placeholders with actual values
    result = template_string
    
    # Customer placeholders
    result = result.replace('{customer.name}', order.customer.name)
    result = result.replace('{customer.email}', order.customer.email)
    result = result.replace('{customer.address}', order.customer.address)
    
    # Order placeholders
    result = result.replace('{order.id}', order.id)
    result = result.replace('{order.total}', f"{order.total:.2f}")
    result = result.replace('{order.items}', ", ".join(order.items))
    result = result.replace('{order.item_count}', str(len(order.items)))
    result = result.replace('{order.payment_method}', order.payment_method)
    
    return result
