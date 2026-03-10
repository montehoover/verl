# Predefined orders storage
orders = {
    "ORD001": {
        "customer_name": "John Smith",
        "total_amount": 150.99
    },
    "ORD002": {
        "customer_name": "Sarah Johnson",
        "total_amount": 89.50
    },
    "ORD003": {
        "customer_name": "Mike Davis",
        "total_amount": 325.00
    }
}

def get_order_details(order_id, format_string=None):
    """
    Retrieves and prints basic order details for a given order ID.
    
    Args:
        order_id (str): The ID of the order to retrieve
        format_string (str, optional): Custom format string with placeholders
                                     like {order_id}, {customer_name}, {total_amount}
    
    Returns:
        str: Formatted string if format_string is provided, None otherwise
    """
    if order_id in orders:
        order = orders[order_id]
        
        if format_string:
            # Create a dictionary with all available placeholders
            format_data = {
                'order_id': order_id,
                'customer_name': order['customer_name'],
                'total_amount': order['total_amount']
            }
            
            # Replace placeholders in format string
            formatted = format_string
            for key, value in format_data.items():
                placeholder = '{' + key + '}'
                if placeholder in formatted:
                    formatted = formatted.replace(placeholder, str(value))
            
            print(formatted)
            return formatted
        else:
            # Default output format
            print(f"Order ID: {order_id}")
            print(f"Customer Name: {order['customer_name']}")
            print(f"Total Amount: ${order['total_amount']:.2f}")
    else:
        error_msg = f"Order {order_id} not found."
        print(error_msg)
        if format_string:
            return error_msg

# Example usage
if __name__ == "__main__":
    # Test the function with different order IDs
    get_order_details("ORD001")
    print()
    get_order_details("ORD002")
    print()
    get_order_details("ORD999")  # Non-existent order
    print()
    
    # Test with custom format strings
    get_order_details("ORD001", "Order #{order_id} - Customer: {customer_name} - Total: ${total_amount}")
    print()
    get_order_details("ORD002", "Customer {customer_name} placed order {order_id}")
    print()
    get_order_details("ORD003", "Invoice for {customer_name}: ${total_amount} (Order: {order_id})")
