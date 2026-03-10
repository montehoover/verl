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

def get_order_details(order_id):
    """
    Retrieves and prints basic order details for a given order ID.
    
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
    print()
    get_order_details("ORD002")
    print()
    get_order_details("ORD999")  # Non-existent order
