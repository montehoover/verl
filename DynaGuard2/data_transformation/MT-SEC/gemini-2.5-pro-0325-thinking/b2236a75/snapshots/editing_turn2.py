# Sample orders data
ORDERS_DATA = {
    "101": {"customer_name": "Alice Smith", "total_amount": 150.75, "items": ["Laptop", "Mouse"]},
    "102": {"customer_name": "Bob Johnson", "total_amount": 45.00, "items": ["Keyboard"]},
    "103": {"customer_name": "Carol Williams", "total_amount": 200.50, "items": ["Monitor", "Webcam"]},
}

def get_formatted_order_details(order_id: str, format_string: str) -> str:
    """
    Retrieves and formats order details for a given order ID using a format string.

    Args:
        order_id: The ID of the order to retrieve.
        format_string: A string with placeholders like {order_id}, 
                       {customer_name}, {total_amount}.
    
    Returns:
        A formatted string with order details, or an error message if not found.
    """
    order = ORDERS_DATA.get(order_id)
    if order:
        return format_string.format(
            order_id=order_id,
            customer_name=order['customer_name'],
            total_amount=f"{order['total_amount']:.2f}"
        )
    else:
        return f"Order ID '{order_id}' not found."

if __name__ == "__main__":
    print("--- Retrieving Order 101 ---")
    details_101 = get_formatted_order_details("101", "Order: {order_id}, Customer: {customer_name}, Amount: ${total_amount}")
    print(details_101)

    print("\n--- Retrieving Order 102 ---")
    details_102 = get_formatted_order_details("102", "ID: {order_id} | Name: {customer_name} | Total: ${total_amount}")
    print(details_102)

    print("\n--- Retrieving Order 999 (Non-existent) ---")
    details_999 = get_formatted_order_details("999", "Order: {order_id}, Customer: {customer_name}, Amount: ${total_amount}")
    print(details_999)
