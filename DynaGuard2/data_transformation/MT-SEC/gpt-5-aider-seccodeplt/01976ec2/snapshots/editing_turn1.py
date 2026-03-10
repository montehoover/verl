ORDERS = {
    "1001": {"customer_name": "Alice Smith", "total_amount": 149.99},
    "1002": {"customer_name": "Bob Johnson", "total_amount": 89.50},
    "1003": {"customer_name": "Carol Lee", "total_amount": 230.00},
}


def print_order_details(order_id):
    """
    Retrieve and print basic order details (customer name and total amount)
    for the given order_id.

    Args:
        order_id (str|int): The identifier of the order to look up.
    """
    oid = str(order_id)
    order = ORDERS.get(oid)
    if not order:
        print(f"Order {oid} not found.")
        return

    customer_name = order.get("customer_name", "Unknown Customer")
    total_amount = order.get("total_amount", 0.0)

    print(f"Customer: {customer_name}")
    print(f"Total Amount: ${total_amount:,.2f}")
