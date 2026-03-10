import string

# Predefined orders data
ORDERS_DATA = {
    "101": {"customer_name": "Alice Smith", "total_amount": 150.75},
    "102": {"customer_name": "Bob Johnson", "total_amount": 89.99},
    "103": {"customer_name": "Carol Williams", "total_amount": 230.50},
    "104": {"customer_name": "David Brown", "total_amount": 45.00},
}

class GracefulFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, f'{{{key}}}') # Return the placeholder itself if key is not found
        else:
            return super().get_value(key, args, kwargs)

def get_formatted_order_details(order_id: str, format_string: str) -> str:
    """
    Retrieves order details and formats them using a provided format string.
    Handles missing placeholders in the format string gracefully by leaving them as is.

    Args:
        order_id: The ID of the order to retrieve.
        format_string: A string with placeholders like {order_id}, {customer_name}, {total_amount}.

    Returns:
        The formatted string with order details, or an error message if the order is not found.
    """
    order = ORDERS_DATA.get(order_id)
    if order:
        formatter = GracefulFormatter()
        data_for_formatting = {
            "order_id": order_id,
            "customer_name": order["customer_name"],
            "total_amount": order["total_amount"] 
        }
        return formatter.format(format_string, **data_for_formatting)
    else:
        return f"Order ID {order_id} not found."

if __name__ == '__main__':
    # Example usage:
    fmt_str1 = "Order {order_id}: Customer - {customer_name}, Amount - ${total_amount:.2f}"
    fmt_str2 = "ID: {order_id}, Name: {customer_name}"
    fmt_str_with_unknown_placeholder = "Order {order_id}, Status: {status}, Amount: ${total_amount:.2f}"

    print("--- Order 101 (Full details) ---")
    print(get_formatted_order_details("101", fmt_str1))
    
    print("\n--- Order 103 (Partial details) ---")
    print(get_formatted_order_details("103", fmt_str2))

    print("\n--- Order 102 (With unknown placeholder {status}) ---")
    print(get_formatted_order_details("102", fmt_str_with_unknown_placeholder))
    
    print("\n--- Order 999 (Non-existent order) ---")
    print(get_formatted_order_details("999", fmt_str1))
