# Sample orders data
ORDERS_DATA = {
    "101": {"customer_name": "Alice Smith", "total_amount": 150.75, "items": ["Laptop", "Mouse"]},
    "102": {"customer_name": "Bob Johnson", "total_amount": 45.00, "items": ["Keyboard"]},
    "103": {"customer_name": "Carol Williams", "total_amount": 200.50, "items": ["Monitor", "Webcam"]},
}

def print_order_details(order_id: str):
    """
    Retrieves and prints basic order details for a given order ID.

    Args:
        order_id: The ID of the order to retrieve.
    """
    order = ORDERS_DATA.get(order_id)
    if order:
        print(f"Order ID: {order_id}")
        print(f"Customer Name: {order['customer_name']}")
        print(f"Total Amount: ${order['total_amount']:.2f}")
    else:
        print(f"Order ID '{order_id}' not found.")

if __name__ == "__main__":
    print("--- Retrieving Order 101 ---")
    print_order_details("101")
    print("\n--- Retrieving Order 102 ---")
    print_order_details("102")
    print("\n--- Retrieving Order 999 (Non-existent) ---")
    print_order_details("999")
