# Predefined orders stored in a dictionary
orders = {
    "ORD001": {
        "customer_name": "John Doe",
        "total_amount": 150.99,
        "items": ["T-shirt", "Jeans"],
        "status": "Shipped"
    },
    "ORD002": {
        "customer_name": "Jane Smith",
        "total_amount": 89.50,
        "items": ["Sneakers"],
        "status": "Processing"
    },
    "ORD003": {
        "customer_name": "Bob Johnson",
        "total_amount": 299.99,
        "items": ["Laptop Case", "Mouse", "Keyboard"],
        "status": "Delivered"
    }
}

def get_order_details(order_id, format_string=None):
    """
    Retrieve and format order details for a given order ID.
    
    Args:
        order_id (str): The order ID to look up
        format_string (str, optional): Template string with placeholders like {order_id}, {customer_name}, etc.
    
    Returns:
        str: Formatted order details or error message
    """
    if order_id in orders:
        order = orders[order_id]
        
        # Create a dictionary with all available order data
        order_data = {
            'order_id': order_id,
            'customer_name': order.get('customer_name', 'N/A'),
            'total_amount': order.get('total_amount', 0.0),
            'status': order.get('status', 'N/A'),
            'items': ', '.join(order.get('items', []))
        }
        
        if format_string:
            try:
                # Replace placeholders with actual values
                formatted_output = format_string.format(**order_data)
                print(formatted_output)
                return formatted_output
            except KeyError as e:
                error_msg = f"Error: Unknown placeholder {e} in format string"
                print(error_msg)
                return error_msg
        else:
            # Default output format
            output = f"Order ID: {order_id}\n"
            output += f"Customer Name: {order_data['customer_name']}\n"
            output += f"Total Amount: ${order_data['total_amount']:.2f}"
            print(output)
            return output
    else:
        error_msg = f"Order {order_id} not found."
        print(error_msg)
        return error_msg

# Example usage
if __name__ == "__main__":
    # Test with default format
    get_order_details("ORD001")
    print()
    
    # Test with custom format string
    get_order_details("ORD002", "Order #{order_id} - Customer: {customer_name} - Total: ${total_amount:.2f}")
    print()
    
    # Test with format string including status and items
    get_order_details("ORD003", "Order {order_id}: {customer_name} ordered {items} (Status: {status})")
    print()
    
    # Test with non-existent order
    get_order_details("ORD999", "Order details: {order_id}")
    print()
    
    # Test with invalid placeholder
    get_order_details("ORD001", "Order {order_id} - {invalid_field}")
