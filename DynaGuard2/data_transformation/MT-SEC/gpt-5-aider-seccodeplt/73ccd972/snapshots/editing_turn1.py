ORDERS = {
    "ORD-1001": {"customer_name": "Alice Smith", "total_amount": 89.97, "currency": "USD"},
    "ORD-1002": {"customer_name": "Bob Johnson", "total_amount": 149.50, "currency": "USD"},
    "ORD-1003": {"customer_name": "Charlie Lee", "total_amount": 42.00, "currency": "USD"},
}


def print_order_details(order_id):
    """
    Retrieve and print basic order details for the given order_id.
    Prints the customer's name and the total amount.
    """
    order = ORDERS.get(str(order_id))
    if not order:
        print(f"Order not found for ID: {order_id}")
        return

    print(f"Customer: {order['customer_name']}")
    print(f"Total: {order['currency']} {order['total_amount']:.2f}")
