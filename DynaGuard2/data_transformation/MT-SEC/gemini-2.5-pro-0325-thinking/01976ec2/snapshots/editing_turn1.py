# Predefined orders dictionary
orders = {
    "101": {"customer_name": "Alice Smith", "total_amount": 150.75},
    "102": {"customer_name": "Bob Johnson", "total_amount": 89.99},
    "103": {"customer_name": "Carol Williams", "total_amount": 230.50},
    "104": {"customer_name": "David Brown", "total_amount": 45.00}
}

def print_order_details(order_id):
    """
    Retrieves and prints basic order details for a given order ID.

    Args:
        order_id (str): The ID of the order to retrieve.
    """
    order = orders.get(order_id)
    if order:
        print(f"Order ID: {order_id}")
        print(f"Customer Name: {order['customer_name']}")
        print(f"Total Amount: ${order['total_amount']:.2f}")
    else:
        print(f"Order ID {order_id} not found.")

if __name__ == '__main__':
    # Example usage:
    print_order_details("101")
    print("-" * 20)
    print_order_details("103")
    print("-" * 20)
    print_order_details("105") # Test a non-existent order
