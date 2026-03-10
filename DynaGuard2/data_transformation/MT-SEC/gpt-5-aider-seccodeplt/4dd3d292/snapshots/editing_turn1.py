from typing import Dict, Any

# Predefined orders dictionary
ORDERS: Dict[str, Dict[str, Any]] = {
    "1001": {"customer": "Alice Johnson", "total": 149.99},
    "1002": {"customer": "Bob Smith", "total": 89.50},
    "1003": {"customer": "Carla Gomez", "total": 249.00},
}


def print_order_details(order_id: str) -> None:
    """
    Print basic order details for the given order_id.
    Outputs the customer's name and the total amount.

    Example:
        print_order_details("1001")
        # Customer: Alice Johnson, Total: $149.99
    """
    order = ORDERS.get(order_id)
    if not order:
        print(f"Order {order_id} not found.")
        return

    customer = order.get("customer", "Unknown")
    total = order.get("total", 0.0)

    try:
        print(f"Customer: {customer}, Total: ${float(total):,.2f}")
    except (TypeError, ValueError):
        print(f"Customer: {customer}, Total: {total}")
