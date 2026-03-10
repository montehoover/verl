class Order:
    def __init__(self, order_id, customer_name, items, total):
        self.order_id = order_id
        self.customer_name = customer_name
        self.items = items
        self.total = total

# Predefined dictionary of invoices
invoices = {
    "ORD-001": Order("ORD-001", "John Doe", ["Laptop", "Mouse"], 1299.99),
    "ORD-002": Order("ORD-002", "Jane Smith", ["Keyboard", "Monitor"], 450.00),
    "ORD-003": Order("ORD-003", "Bob Johnson", ["Webcam"], 89.99),
}

def get_order_by_id(order_id):
    if order_id in invoices:
        return invoices[order_id]
    else:
        raise ValueError(f"Order ID '{order_id}' not found")
