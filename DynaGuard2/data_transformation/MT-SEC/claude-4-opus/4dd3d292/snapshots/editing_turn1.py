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

def get_order_details(order_id):
    """
    Retrieve and print basic order details for a given order ID.
    
    Args:
        order_id (str): The order ID to look up
    """
    if order_id in orders:
        order = orders[order_id]
        print(f"Order ID: {order_id}")
        print(f"Customer Name: {order['customer_name']}")
        print(f"Total Amount: ${order['total_amount']:.2f}")
    else:
        print(f"Order {order_id} not found.")

# Example usage
if __name__ == "__main__":
    # Test the function with different order IDs
    get_order_details("ORD001")
    print()
    get_order_details("ORD002")
    print()
    get_order_details("ORD999")  # Non-existent order
