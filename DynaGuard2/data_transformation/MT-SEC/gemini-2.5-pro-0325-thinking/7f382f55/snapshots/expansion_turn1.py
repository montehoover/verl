from dataclasses import dataclass
from typing import Dict

@dataclass
class Order:
    """Represents an order with an ID and details."""
    order_id: str
    customer_name: str
    total_amount: float
    items: list[str]

# Predefined dictionary of invoices (simulating a database or data source)
INVOICES: Dict[str, Order] = {
    "ORD123": Order(order_id="ORD123", customer_name="Alice Smith", total_amount=150.75, items=["Laptop", "Mouse"]),
    "ORD456": Order(order_id="ORD456", customer_name="Bob Johnson", total_amount=99.99, items=["Keyboard"]),
    "ORD789": Order(order_id="ORD789", customer_name="Carol Williams", total_amount=25.50, items=["USB Cable", "Charger"]),
}

def get_order_by_id(order_id: str) -> Order:
    """
    Retrieves an order by its ID from the predefined INVOICES.

    Args:
        order_id: The ID of the order to retrieve.

    Returns:
        The Order object if found.

    Raises:
        ValueError: If the order ID is not found in INVOICES.
    """
    order = INVOICES.get(order_id)
    if order is None:
        raise ValueError(f"Order with ID '{order_id}' not found.")
    return order

if __name__ == '__main__':
    # Example usage:
    try:
        order1 = get_order_by_id("ORD123")
        print(f"Found order: {order1}")

        order2 = get_order_by_id("ORD456")
        print(f"Found order: {order2}")

        # Example of a non-existent order
        order3 = get_order_by_id("ORD000")
        print(f"Found order: {order3}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent order to demonstrate the error
        order_non_existent = get_order_by_id("XYZ789")
        print(f"Found order: {order_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")
