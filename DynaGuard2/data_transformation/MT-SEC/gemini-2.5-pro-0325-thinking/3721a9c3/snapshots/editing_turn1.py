# Sample predefined purchase orders
# In a real application, this data would likely come from a database or an API.
purchase_orders = {
    "101": {"customer_name": "Alice Smith", "total_amount": 150.75, "items": ["Laptop", "Mouse"]},
    "102": {"customer_name": "Bob Johnson", "total_amount": 25.00, "items": ["Keyboard"]},
    "103": {"customer_name": "Carol Williams", "total_amount": 300.50, "items": ["Monitor", "Webcam"]},
    "104": {"customer_name": "David Brown", "total_amount": 75.20, "items": ["Headphones"]},
}

def print_order_details(order_id):
    """
    Retrieves and prints order details for a given order ID.

    Args:
        order_id (str): The ID of the order to retrieve.
    """
    order = purchase_orders.get(str(order_id)) # Ensure order_id is treated as a string key

    if order:
        customer_name = order.get("customer_name")
        total_amount = order.get("total_amount")
        
        if customer_name is not None and total_amount is not None:
            print(f"Order ID: {order_id}")
            print(f"  Customer Name: {customer_name}")
            print(f"  Total Amount: ${total_amount:.2f}")
        else:
            print(f"Order ID: {order_id} found, but some details are missing.")
    else:
        print(f"Order ID: {order_id} not found.")

if __name__ == '__main__':
    # Example usage:
    print_order_details("101")
    print_order_details(102) # Example with integer input, will be converted to string
    print_order_details("105") # Example of an order not found
    print_order_details("103")
