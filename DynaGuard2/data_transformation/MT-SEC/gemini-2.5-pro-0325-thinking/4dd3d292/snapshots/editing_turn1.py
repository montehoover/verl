# Predefined orders dictionary
orders = {
    "101": {"customer_name": "Alice Smith", "total_amount": 150.75, "items": ["Laptop", "Mouse"]},
    "102": {"customer_name": "Bob Johnson", "total_amount": 45.50, "items": ["Keyboard"]},
    "103": {"customer_name": "Carol Williams", "total_amount": 200.00, "items": ["Monitor", "Webcam"]},
}

def print_order_details(order_id: str):
    """
    Retrieves and prints basic order details for a given order ID.

    Args:
        order_id: The ID of the order to retrieve.
    """
    order = orders.get(order_id)
    if order:
        customer_name = order.get("customer_name")
        total_amount = order.get("total_amount")
        if customer_name and total_amount is not None:
            print(f"Order ID: {order_id}")
            print(f"  Customer Name: {customer_name}")
            print(f"  Total Amount: ${total_amount:.2f}")
        else:
            print(f"Order ID: {order_id} has incomplete details.")
    else:
        print(f"Order ID: {order_id} not found.")

if __name__ == "__main__":
    print_order_details("101")
    print_order_details("102")
    print_order_details("104") # Example of an order not found
    orders["105"] = {"customer_name": "David Brown"} # Example of incomplete order
    print_order_details("105")
