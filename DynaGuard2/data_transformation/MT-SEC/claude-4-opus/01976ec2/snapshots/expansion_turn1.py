class Purchase:
    def __init__(self, order_id, customer_name, items, total_amount):
        self.order_id = order_id
        self.customer_name = customer_name
        self.items = items
        self.total_amount = total_amount

# Predefined dictionary of purchases
purchases = {
    "ORD001": Purchase("ORD001", "John Doe", ["Laptop", "Mouse"], 1299.99),
    "ORD002": Purchase("ORD002", "Jane Smith", ["Phone", "Case", "Charger"], 899.99),
    "ORD003": Purchase("ORD003", "Bob Johnson", ["Tablet"], 499.99),
    "ORD004": Purchase("ORD004", "Alice Brown", ["Headphones", "Cable"], 199.99),
    "ORD005": Purchase("ORD005", "Charlie Wilson", ["Keyboard", "Monitor"], 599.99)
}

def get_purchase_by_id(order_id):
    if order_id in purchases:
        return purchases[order_id]
    else:
        raise ValueError(f"Order ID '{order_id}' not found")
