from typing import Dict, Any
import sys

# Predefined orders store (in-memory example)
ORDERS: Dict[str, Dict[str, Any]] = {
    "1001": {"customer_name": "Alice Johnson", "total": 149.97},
    "1002": {"customer_name": "Bob Smith", "total": 89.50},
    "1003": {"customer_name": "Carlos Diaz", "total": 320.00},
    "INV-2001": {"customer_name": "Dana Lee", "total": 45.25},
}


def print_order_details(order_id):
    """
    Retrieve and print basic order details (customer name and total amount)
    for the given order_id. If the order does not exist, prints a not-found message.

    Args:
        order_id: The identifier for the order (string or number).
    """
    oid = str(order_id)
    order = ORDERS.get(oid)

    if not order:
        print(f"Order {oid} not found.")
        return

    customer_name = order.get("customer_name", "Unknown Customer")
    total = order.get("total", 0.0)

    print(f"Customer: {customer_name}")
    print(f"Total Amount: ${total:.2f}")


if __name__ == "__main__":
    # Optional CLI usage: python multiturn-editing.py <order_id>
    if len(sys.argv) > 1:
        print_order_details(sys.argv[1])
    else:
        print("Usage: python multiturn-editing.py <order_id>")
