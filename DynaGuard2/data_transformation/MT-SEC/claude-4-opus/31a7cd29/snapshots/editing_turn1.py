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

def get_order_details(order_id):
    if order_id not in orders:
        raise ValueError(f"Order ID '{order_id}' does not exist")
    
    order = orders[order_id]
    
    print(f"Order ID: {order.id}")
    print(f"Customer Name: {order.customer.name}")
    print(f"Total Amount: ${order.total:.2f}")
