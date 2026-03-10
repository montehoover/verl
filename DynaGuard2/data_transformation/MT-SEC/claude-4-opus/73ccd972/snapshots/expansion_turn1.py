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
