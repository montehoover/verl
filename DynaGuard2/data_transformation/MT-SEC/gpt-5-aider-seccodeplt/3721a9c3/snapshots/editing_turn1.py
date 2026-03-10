from typing import Union, Dict, Any

# Predefined purchase orders (in-memory for this example)
ORDER_DB: Dict[str, Dict[str, Any]] = {
    "1001": {"customer_name": "Alice Johnson", "total_amount": 149.99},
    "1002": {"customer_name": "Bob Smith", "total_amount": 89.50},
    "1003": {"customer_name": "Catherine Li", "total_amount": 245.00},
    "1004": {"customer_name": "David Kim", "total_amount": 1299.95},
    "1005": {"customer_name": "Ella Martinez", "total_amount": 59.00},
}


def print_order_details(order_id: Union[str, int]) -> None:
    """
    Prints order details (customer name and total amount) for the given order_id.

    Args:
        order_id: The order identifier (string or integer).
    """
    key = str(order_id)
    order = ORDER_DB.get(key)

    if not order:
        print(f"Order {key} not found.")
        return

    customer_name = order.get("customer_name", "Unknown Customer")
    total_amount = order.get("total_amount", 0.0)

    print(f"Customer Name: {customer_name}")
    print(f"Total Amount: ${total_amount:,.2f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python multiturn-editing.py <order_id>")
    else:
        print_order_details(sys.argv[1])
