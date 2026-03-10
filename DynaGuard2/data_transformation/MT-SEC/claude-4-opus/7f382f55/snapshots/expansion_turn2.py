import re

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

def check_template_placeholders(template):
    # Define valid placeholders
    valid_placeholders = [
        '{customer.full_name}',
        '{customer.email}',
        '{customer.phone}',
        '{customer.address}',
        '{order.order_id}',
        '{order.date}',
        '{order.total_amount}',
        '{order.subtotal}',
        '{order.tax}',
        '{order.shipping}',
        '{order.items}',
        '{order.quantity}',
        '{order.status}',
        '{company.name}',
        '{company.address}',
        '{company.phone}',
        '{company.email}'
    ]
    
    # Find all placeholders in the template
    placeholders = re.findall(r'\{[^}]+\}', template)
    
    # Check each placeholder
    for placeholder in placeholders:
        if placeholder not in valid_placeholders:
            raise ValueError(f"Invalid placeholder: {placeholder}")
    
    return True
