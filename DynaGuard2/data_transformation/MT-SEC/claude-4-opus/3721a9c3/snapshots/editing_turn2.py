# Sample order data (in a real application, this would come from a database)
orders = {
    "ORD001": {
        "customer_name": "John Smith",
        "total_amount": 129.99,
        "items": ["T-shirt", "Jeans"],
        "status": "Delivered"
    },
    "ORD002": {
        "customer_name": "Jane Doe",
        "total_amount": 89.50,
        "items": ["Shoes"],
        "status": "Processing"
    },
    "ORD003": {
        "customer_name": "Bob Johnson",
        "total_amount": 245.00,
        "items": ["Laptop Case", "Mouse", "Keyboard"],
        "status": "Shipped"
    }
}

def get_order_details(order_id, format_template=None):
    """
    Retrieves and formats order details based on order ID and template.
    
    Args:
        order_id (str): The ID of the order to retrieve
        format_template (str, optional): Template string with placeholders like {order_id}, {customer_name}, etc.
    
    Returns:
        str: Formatted order details or error message
    """
    if order_id not in orders:
        return f"Order {order_id} not found."
    
    order = orders[order_id]
    
    # Default template if none provided
    if format_template is None:
        format_template = "Order ID: {order_id}\nCustomer Name: {customer_name}\nTotal Amount: ${total_amount:.2f}"
    
    # Prepare data for formatting
    format_data = {
        'order_id': order_id,
        'customer_name': order['customer_name'],
        'total_amount': order['total_amount'],
        'status': order['status'],
        'items': ', '.join(order['items']),
        'item_count': len(order['items'])
    }
    
    # Replace placeholders with actual values
    result = format_template
    for key, value in format_data.items():
        placeholder = '{' + key + '}'
        if placeholder in result:
            # Handle float formatting for total_amount
            if key == 'total_amount':
                # Check if the placeholder has formatting
                formatted_placeholder = '{' + key + ':.2f}'
                if formatted_placeholder in result:
                    result = result.replace(formatted_placeholder, f"{value:.2f}")
                else:
                    result = result.replace(placeholder, str(value))
            else:
                result = result.replace(placeholder, str(value))
    
    return result

# Example usage
if __name__ == "__main__":
    # Test with default template
    print(get_order_details("ORD001"))
    print("-" * 50)
    
    # Test with custom template
    custom_template = "Order #{order_id} - {customer_name} - ${total_amount:.2f} ({status})"
    print(get_order_details("ORD002", custom_template))
    print("-" * 50)
    
    # Test with template including items
    detailed_template = """
Order Details:
- ID: {order_id}
- Customer: {customer_name}
- Items ({item_count}): {items}
- Total: ${total_amount:.2f}
- Status: {status}
"""
    print(get_order_details("ORD003", detailed_template))
    print("-" * 50)
    
    # Test with non-existent order
    print(get_order_details("ORD999", custom_template))
    print("-" * 50)
    
    # Test with template containing unknown placeholders
    template_with_unknown = "Order {order_id} by {customer_name} - {unknown_field}"
    print(get_order_details("ORD001", template_with_unknown))
