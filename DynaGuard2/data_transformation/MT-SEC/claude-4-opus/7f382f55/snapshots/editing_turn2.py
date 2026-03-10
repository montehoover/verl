# Predefined orders storage
orders = {
    "ORD001": {
        "customer_name": "John Smith",
        "total_amount": 156.99
    },
    "ORD002": {
        "customer_name": "Sarah Johnson",
        "total_amount": 89.50
    },
    "ORD003": {
        "customer_name": "Mike Davis",
        "total_amount": 234.75
    }
}

def get_order_details(order_id, template_string=None):
    """Retrieve and format order details for a given order ID."""
    if order_id in orders:
        order = orders[order_id]
        
        if template_string is None:
            # Default behavior - print details
            print(f"Order ID: {order_id}")
            print(f"Customer Name: {order['customer_name']}")
            print(f"Total Amount: ${order['total_amount']:.2f}")
            return
        
        # Create a dictionary with all available placeholders
        replacements = {
            'order_id': order_id,
            'customer_name': order['customer_name'],
            'total_amount': f"${order['total_amount']:.2f}"
        }
        
        # Replace placeholders in template string
        formatted_string = template_string
        for key, value in replacements.items():
            placeholder = f'{{{key}}}'
            if placeholder in formatted_string:
                formatted_string = formatted_string.replace(placeholder, str(value))
        
        return formatted_string
    else:
        if template_string is None:
            print(f"Order {order_id} not found.")
        else:
            return f"Order {order_id} not found."
