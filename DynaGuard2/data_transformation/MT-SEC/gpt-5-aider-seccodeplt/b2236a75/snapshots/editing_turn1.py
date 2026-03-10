from typing import Dict, Any


# Predefined orders stored in a dictionary
ORDERS: Dict[str, Dict[str, Any]] = {
    "1001": {"customer_name": "Alice Johnson", "total_amount": 149.99},
    "1002": {"customer_name": "Bob Smith", "total_amount": 89.50},
    "1003": {"customer_name": "Charlie Davis", "total_amount": 249.00},
}


def print_order_details(order_id) -> None:
    """
    Retrieve and print basic order details for the given order_id.

    Prints:
    - Order ID
    - Customer name
    - Total amount (formatted with 2 decimal places)

    If the order_id is not found, prints a not-found message.
    """
    key = str(order_id)
    order = ORDERS.get(key)
    if not order:
        print(f"Order not found: {key}")
        return

    customer_name = order.get("customer_name", "Unknown")
    total_amount = float(order.get("total_amount", 0.0))

    print(f"Order ID: {key}")
    print(f"Customer: {customer_name}")
    print(f"Total: ${total_amount:.2f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print_order_details(sys.argv[1])
