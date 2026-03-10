# Predefined orders data
PREDEFINED_ORDERS = {
    "101": {"customer_name": "Alice Smith", "total_amount": 150.75},
    "102": {"customer_name": "Bob Johnson", "total_amount": 89.99},
    "103": {"customer_name": "Carol Williams", "total_amount": 230.50},
}

def print_order_details(order_id: str):
    """
    Retrieves and prints basic order details for a given order ID.

    Args:
        order_id: The ID of the order to retrieve.
    """
    order = PREDEFINED_ORDERS.get(order_id)
    if order:
        print(f"Order ID: {order_id}")
        print(f"Customer Name: {order['customer_name']}")
        print(f"Total Amount: ${order['total_amount']:.2f}")
    else:
        print(f"Order ID {order_id} not found.")

if __name__ == "__main__":
    # Example usage:
    print_order_details("101")
    print("-" * 20)
    print_order_details("102")
    print("-" * 20)
    print_order_details("999") # Test a non-existent order
