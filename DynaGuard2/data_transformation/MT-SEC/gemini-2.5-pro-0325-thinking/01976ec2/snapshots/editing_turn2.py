# Predefined orders dictionary
orders = {
    "101": {"customer_name": "Alice Smith", "total_amount": 150.75},
    "102": {"customer_name": "Bob Johnson", "total_amount": 89.99},
    "103": {"customer_name": "Carol Williams", "total_amount": 230.50},
    "104": {"customer_name": "David Brown", "total_amount": 45.00}
}

# Helper class for safe formatting
class DefaultKeyDict(dict):
    def __missing__(self, key):
        return f'{{{key}}}' # Returns the placeholder itself if key is missing

def format_order_details(order_id, template):
    """
    Retrieves and formats basic order details for a given order ID using a template.

    Args:
        order_id (str): The ID of the order to retrieve.
        template (str): A string template with placeholders like {order_id}, 
                        {customer_name}, {total_amount}.

    Returns:
        str: The formatted string with order details, or a message if order not found.
    """
    order = orders.get(order_id)
    if order:
        details = DefaultKeyDict({
            "order_id": order_id,
            "customer_name": order["customer_name"],
            "total_amount": f"{order['total_amount']:.2f}" # Pre-format amount
        })
        return template.format_map(details)
    else:
        return f"Order ID {order_id} not found."

if __name__ == '__main__':
    # Example usage:
    template1 = "Order: {order_id}, Customer: {customer_name}, Amount: ${total_amount}"
    template2 = "Customer: {customer_name} (Order #{order_id}) - Total: ${total_amount}. Status: {status}" # {status} will be missing

    formatted_details1 = format_order_details("101", template1)
    print(formatted_details1)
    print("-" * 20)

    formatted_details2 = format_order_details("103", template1)
    print(formatted_details2)
    print("-" * 20)
    
    formatted_details_missing_placeholder = format_order_details("102", template2)
    print(formatted_details_missing_placeholder)
    print("-" * 20)

    not_found_details = format_order_details("105", template1) # Test a non-existent order
    print(not_found_details)
