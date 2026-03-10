class Order:
    def __init__(self, order_id, customer_name, items, total):
        self.order_id = order_id
        self.customer_name = customer_name
        self.items = items
        self.total = total

# Predefined dictionary of orders
orders = {
    "ORD001": Order("ORD001", "John Doe", ["Laptop", "Mouse"], 1299.99),
    "ORD002": Order("ORD002", "Jane Smith", ["Keyboard", "Monitor"], 459.98),
    "ORD003": Order("ORD003", "Bob Johnson", ["Headphones"], 89.99),
}

def get_order_by_id(order_id):
    """
    Fetch order details by order ID.
    
    Args:
        order_id (str): The ID of the order to retrieve
        
    Returns:
        Order: The Order object corresponding to the given ID
        
    Raises:
        ValueError: If the order ID is not found
    """
    if order_id in orders:
        return orders[order_id]
    else:
        raise ValueError(f"Order with ID '{order_id}' not found")
