from typing import Dict, NamedTuple

# Define a simple Order structure (can be a class or NamedTuple)
class Order(NamedTuple):
    order_id: str
    customer_name: str
    total_amount: float
    items: list[str]

# Predefined dictionary of orders
# In a real application, this would come from a database or an API
_ORDERS_DATABASE: Dict[str, Order] = {
    "order_123": Order(order_id="order_123", customer_name="Alice Smith", total_amount=150.75, items=["item_A", "item_B"]),
    "order_456": Order(order_id="order_456", customer_name="Bob Johnson", total_amount=99.99, items=["item_C"]),
    "order_789": Order(order_id="order_789", customer_name="Carol Williams", total_amount=230.00, items=["item_A", "item_D", "item_E"]),
}

def get_order_by_id(order_id: str) -> Order:
    """
    Fetches order details by order ID.

    Args:
        order_id: The ID of the order to fetch.

    Returns:
        The Order object if found.

    Raises:
        ValueError: If the order ID is not found in the database.
    """
    order = _ORDERS_DATABASE.get(order_id)
    if order is None:
        raise ValueError(f"Order with ID '{order_id}' not found.")
    return order

if __name__ == '__main__':
    # Example usage:
    try:
        order1 = get_order_by_id("order_123")
        print(f"Found order: {order1}")

        order2 = get_order_by_id("order_456")
        print(f"Found order: {order2}")

        # Example of a non-existent order
        order_non_existent = get_order_by_id("order_000")
        print(f"Found order: {order_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example demonstrating another non-existent order
    try:
        order_another_non_existent = get_order_by_id("order_xyz")
        print(f"Found order: {order_another_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")
