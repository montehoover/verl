# Predefined orders data
PREDEFINED_ORDERS = {
    "101": {"customer_name": "Alice Smith", "total_amount": 150.75},
    "102": {"customer_name": "Bob Johnson", "total_amount": 89.99},
    "103": {"customer_name": "Carol Williams", "total_amount": 230.50},
}

class SafeFormatter(dict):
    """
    A dictionary subclass that returns the placeholder itself (e.g., '{key}')
    if a key is not found. Used with str.format_map().
    """
    def __missing__(self, key):
        return f"{{{key}}}"

def get_formatted_order_details(order_id: str, template_string: str) -> str:
    """
    Retrieves order details and formats them using a template string.

    Args:
        order_id: The ID of the order to retrieve.
        template_string: A string with placeholders like {order_id},
                         {customer_name}, {total_amount}. Other placeholders
                         will be returned as is (e.g., {unknown_placeholder}).

    Returns:
        A formatted string with order details, or an error message
        if the order is not found.
    """
    order = PREDEFINED_ORDERS.get(order_id)
    if order:
        details = {
            "order_id": order_id,
            "customer_name": order["customer_name"],
            "total_amount": f"{order['total_amount']:.2f}"  # Pre-format amount
        }
        # Use SafeFormatter to handle placeholders in template_string
        # that are not in 'details' gracefully.
        return template_string.format_map(SafeFormatter(details))
    else:
        return f"Order ID {order_id} not found."

if __name__ == "__main__":
    # Example usage:
    template1 = "Order: {order_id}, Customer: {customer_name}, Amount: ${total_amount}."
    template2 = "Customer {customer_name} - Order {order_id}."
    template_with_missing_placeholder = "Order {order_id}, Status: {status}, Amount: ${total_amount}"

    print("--- Order 101 (Template 1) ---")
    formatted_details_101 = get_formatted_order_details("101", template1)
    print(formatted_details_101)
    
    print("\n--- Order 102 (Template 2) ---")
    formatted_details_102 = get_formatted_order_details("102", template2)
    print(formatted_details_102)
    
    print("\n--- Order 103 (Template with missing placeholder) ---")
    formatted_details_103 = get_formatted_order_details("103", template_with_missing_placeholder)
    print(formatted_details_103)

    print("\n--- Non-existent Order 999 (Template 1) ---")
    formatted_details_999 = get_formatted_order_details("999", template1)
    print(formatted_details_999)
