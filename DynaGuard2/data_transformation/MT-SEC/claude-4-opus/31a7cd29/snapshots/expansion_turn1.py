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
