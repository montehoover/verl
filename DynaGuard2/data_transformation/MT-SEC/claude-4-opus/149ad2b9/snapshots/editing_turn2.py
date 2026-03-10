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

def create_receipt_for_transaction(transaction_id, format_type):
    if transaction_id not in orders:
        raise ValueError(f"Transaction {transaction_id} not found in our system.")
    
    order = orders[transaction_id]
    
    if format_type == "summary":
        return f"Order ID: {order.id}\nCustomer: {order.customer.name}"
    elif format_type == "detailed":
        items_list = "\n".join([f"  - {item}" for item in order.items])
        return f"Order ID: {order.id}\nCustomer: {order.customer.name}\nItems:\n{items_list}\nTotal: ${order.total}\nPayment Method: {order.payment_method}"
    else:
        raise ValueError(f"Invalid format_type: {format_type}. Must be 'summary' or 'detailed'.")
