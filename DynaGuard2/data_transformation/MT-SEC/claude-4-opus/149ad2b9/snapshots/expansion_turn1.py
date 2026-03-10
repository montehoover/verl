class Order:
    def __init__(self, transaction_id, customer_name, items, total):
        self.transaction_id = transaction_id
        self.customer_name = customer_name
        self.items = items
        self.total = total

# Example dictionary of orders (would be populated with actual data)
orders = {
    "TXN001": Order("TXN001", "John Doe", ["Item1", "Item2"], 99.99),
    "TXN002": Order("TXN002", "Jane Smith", ["Item3"], 49.99),
    "TXN003": Order("TXN003", "Bob Johnson", ["Item4", "Item5", "Item6"], 199.99)
}

def get_order_by_id(transaction_id):
    """
    Retrieves an Order object by transaction ID.
    
    Args:
        transaction_id: The transaction ID to look up
        
    Returns:
        Order object corresponding to the transaction ID
        
    Raises:
        ValueError: If transaction_id is None, empty, or not found in orders
    """
    if not transaction_id:
        raise ValueError("Transaction ID cannot be None or empty")
    
    if transaction_id not in orders:
        raise ValueError(f"Transaction ID '{transaction_id}' not found")
    
    return orders[transaction_id]
