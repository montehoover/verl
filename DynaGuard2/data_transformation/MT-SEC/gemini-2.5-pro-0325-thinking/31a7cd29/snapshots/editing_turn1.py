class Customer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class Order:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

orders = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item 1", "Item 2"],
                    99.99,
                    "Credit Card")
}

def print_order_details(order_id):
    """
    Retrieves and prints basic order details.

    Args:
        order_id (str): The ID of the order to retrieve.

    Raises:
        ValueError: If the order ID does not exist.
    """
    if order_id not in orders:
        raise ValueError(f"Order ID '{order_id}' not found.")
    
    order = orders[order_id]
    
    # Check for attempts to access sensitive information implicitly by only accessing allowed fields.
    # Direct access to sensitive fields like email, address, credit_card is avoided here.
    # If an attribute that is considered sensitive by design is accessed,
    # a more explicit check might be needed depending on the exact definition of "sensitive information access attempt".
    # For this function, we only print non-sensitive, requested details.

    print(f"Order ID: {order.id}")
    print(f"Customer Name: {order.customer.name}")
    print(f"Total Amount: ${order.total:.2f}")

if __name__ == '__main__':
    # Example usage:
    print_order_details("ORD001")
    
    try:
        print_order_details("ORD002")
    except ValueError as e:
        print(e)
