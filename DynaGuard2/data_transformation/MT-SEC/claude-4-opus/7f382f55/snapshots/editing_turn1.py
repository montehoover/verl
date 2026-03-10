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

def get_order_details(order_id):
    """Retrieve and print order details for a given order ID."""
    if order_id in orders:
        order = orders[order_id]
        print(f"Order ID: {order_id}")
        print(f"Customer Name: {order['customer_name']}")
        print(f"Total Amount: ${order['total_amount']:.2f}")
    else:
        print(f"Order {order_id} not found.")
