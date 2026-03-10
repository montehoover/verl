from typing import Dict, Any

# Placeholder for the Order class.
# In a real application, this would be a more complex object.
class Order:
    def __init__(self, order_id: str, details: Dict[str, Any]):
        self.order_id = order_id
        self.details = details

    def __repr__(self) -> str:
        return f"Order(order_id='{self.order_id}', details={self.details})"

# Sample orders database (dictionary)
# In a real application, this would come from a database or other persistent storage.
SAMPLE_ORDERS_DB: Dict[str, Order] = {
    "txn_123": Order(order_id="order_abc", details={"item": "Laptop", "quantity": 1, "price": 1200.00}),
    "txn_456": Order(order_id="order_def", details={"item": "Mouse", "quantity": 2, "price": 25.00}),
    "txn_789": Order(order_id="order_ghi", details={"item": "Keyboard", "quantity": 1, "price": 75.00}),
}

def get_order_by_id(transaction_id: str, orders_db: Dict[str, Order]) -> Order:
    """
    Retrieves an Order object from a dictionary of orders using the transaction ID.

    Args:
        transaction_id: The ID of the transaction to retrieve.
        orders_db: A dictionary where keys are transaction IDs and values are Order objects.

    Returns:
        The Order object corresponding to the transaction ID.

    Raises:
        ValueError: If the transaction ID is invalid (e.g., not a string, empty)
                    or not found in the orders_db.
    """
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ValueError("Transaction ID must be a non-empty string.")

    order = orders_db.get(transaction_id)
    if order is None:
        raise ValueError(f"Order with transaction ID '{transaction_id}' not found.")
    
    return order

if __name__ == '__main__':
    # Example Usage:
    print("Available orders:", SAMPLE_ORDERS_DB)

    # Test case 1: Valid transaction ID
    try:
        order1 = get_order_by_id("txn_123", SAMPLE_ORDERS_DB)
        print(f"\nFound order for txn_123: {order1}")
    except ValueError as e:
        print(f"\nError for txn_123: {e}")

    # Test case 2: Another valid transaction ID
    try:
        order2 = get_order_by_id("txn_456", SAMPLE_ORDERS_DB)
        print(f"\nFound order for txn_456: {order2}")
    except ValueError as e:
        print(f"\nError for txn_456: {e}")

    # Test case 3: Invalid transaction ID (not found)
    try:
        order_not_found = get_order_by_id("txn_000", SAMPLE_ORDERS_DB)
        print(f"\nFound order for txn_000: {order_not_found}")
    except ValueError as e:
        print(f"\nError for txn_000: {e}")

    # Test case 4: Invalid transaction ID (empty string)
    try:
        order_empty_id = get_order_by_id("", SAMPLE_ORDERS_DB)
        print(f"\nFound order for empty ID: {order_empty_id}")
    except ValueError as e:
        print(f"\nError for empty ID: {e}")

    # Test case 5: Invalid transaction ID (wrong type)
    try:
        order_wrong_type = get_order_by_id(123, SAMPLE_ORDERS_DB) # type: ignore
        print(f"\nFound order for wrong type ID: {order_wrong_type}")
    except ValueError as e:
        print(f"\nError for wrong type ID: {e}")
