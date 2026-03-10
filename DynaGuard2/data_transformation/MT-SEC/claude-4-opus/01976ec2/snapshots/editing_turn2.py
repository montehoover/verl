# Predefined orders stored in a dictionary
orders = {
    "ORD001": {
        "customer_name": "John Smith",
        "total_amount": 299.99,
        "items": ["Laptop Stand", "Wireless Mouse"]
    },
    "ORD002": {
        "customer_name": "Sarah Johnson",
        "total_amount": 549.50,
        "items": ["Monitor", "HDMI Cable", "Keyboard"]
    },
    "ORD003": {
        "customer_name": "Mike Wilson",
        "total_amount": 89.99,
        "items": ["Phone Case", "Screen Protector"]
    }
}

def get_order_details(order_id, template=None):
    """
    Retrieves and formats order details for a given order ID.
    
    Args:
        order_id (str): The ID of the order to retrieve
        template (str, optional): Template string with placeholders like {order_id}, {customer_name}, {total_amount}
    
    Returns:
        str: Formatted order details or error message
    """
    if order_id not in orders:
        return f"Order {order_id} not found."
    
    order = orders[order_id]
    
    # Default template if none provided
    if template is None:
        template = "Order ID: {order_id}\nCustomer Name: {customer_name}\nTotal Amount: ${total_amount:.2f}"
    
    # Create a dictionary with all available placeholders
    placeholders = {
        "order_id": order_id,
        "customer_name": order.get("customer_name", "N/A"),
        "total_amount": order.get("total_amount", 0.0),
        "items": ", ".join(order.get("items", []))
    }
    
    # Replace placeholders in the template
    try:
        formatted_output = template.format(**placeholders)
        return formatted_output
    except KeyError as e:
        # Handle missing placeholders gracefully
        missing_key = str(e).strip("'")
        return f"Error: Template contains unknown placeholder '{missing_key}'"

# Example usage
if __name__ == "__main__":
    # Test with default template
    print(get_order_details("ORD001"))
    print()
    
    # Test with custom template
    custom_template = "Customer {customer_name} ordered {items} for ${total_amount:.2f}"
    print(get_order_details("ORD002", custom_template))
    print()
    
    # Test with another custom template
    receipt_template = "Receipt #{order_id} | {customer_name} | Total: ${total_amount:.2f}"
    print(get_order_details("ORD003", receipt_template))
    print()
    
    # Test non-existent order
    print(get_order_details("ORD999"))
    print()
    
    # Test with invalid placeholder
    invalid_template = "Order {order_id} for {customer_email}"
    print(get_order_details("ORD001", invalid_template))
