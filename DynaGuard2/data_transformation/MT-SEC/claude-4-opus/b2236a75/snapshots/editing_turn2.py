# E-commerce order details retrieval system

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
    Retrieves and formats order details for a given order ID.
    
    Args:
        order_id (str): The unique identifier for the order
        format_string (str): Optional format string with placeholders like {order_id}, {customer_name}, {total_amount}, {status}
    
    Returns:
        str: Formatted order details or error message
    """
    if order_id in orders:
        order = orders[order_id]
        
        if format_string:
            # Create a dictionary with all available placeholders
            format_data = {
                "order_id": order_id,
                "customer_name": order["customer_name"],
                "total_amount": order["total_amount"],
                "status": order["status"]
            }
            
            # Replace placeholders in the format string
            formatted_output = format_string.format(**format_data)
            print(formatted_output)
            return formatted_output
        else:
            # Default format
            output = f"Order ID: {order_id}\n"
            output += f"Customer Name: {order['customer_name']}\n"
            output += f"Total Amount: ${order['total_amount']:.2f}"
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
    print("-" * 30)
    
    # Test with custom format string
    get_order_details("ORD002", "Order #{order_id} for {customer_name} - Total: ${total_amount:.2f}")
    print("-" * 30)
    
    # Test with another custom format
    get_order_details("ORD003", "{customer_name}'s order ({order_id}) is {status}. Amount: ${total_amount}")
    print("-" * 30)
    
    # Test with non-existent order
    get_order_details("ORD999")
