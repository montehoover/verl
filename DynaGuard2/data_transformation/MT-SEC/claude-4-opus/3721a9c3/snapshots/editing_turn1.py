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

def get_order_details(order_id):
    """
    Retrieves and prints order details based on order ID.
    
    Args:
        order_id (str): The ID of the order to retrieve
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
    print("-" * 30)
    get_order_details("ORD002")
    print("-" * 30)
    get_order_details("ORD999")  # Non-existent order
