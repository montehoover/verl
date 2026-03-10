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
