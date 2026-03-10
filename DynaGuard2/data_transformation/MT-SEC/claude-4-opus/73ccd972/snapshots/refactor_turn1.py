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

def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    # Check if order exists
    if purchase_id not in orders:
        raise ValueError(f"Order ID {purchase_id} does not exist")
    
    order = orders[purchase_id]
    
    # Define restricted/protected fields
    restricted_fields = {'credit_card', 'creditcard', 'credit-card', 'cc'}
    
    # Check for restricted fields in template
    import re
    placeholders = re.findall(r'\{([^}]+)\}', receipt_template)
    
    for placeholder in placeholders:
        # Check if placeholder tries to access credit card info
        if any(restricted in placeholder.lower() for restricted in restricted_fields):
            raise ValueError(f"Access to restricted customer data is not allowed: {placeholder}")
    
    # Create replacement dictionary
    replacements = {
        'order.id': order.id,
        'order.total': str(order.total),
        'order.payment_method': order.payment_method,
        'order.items': ', '.join(order.items),
        'customer.name': order.customer.name,
        'customer.email': order.customer.email,
        'customer.address': order.customer.address
    }
    
    # Replace placeholders in template
    result = receipt_template
    for placeholder in placeholders:
        key = placeholder.strip()
        if key in replacements:
            result = result.replace(f'{{{placeholder}}}', replacements[key])
        else:
            # Check if it's a valid attribute access
            parts = key.split('.')
            if len(parts) == 2:
                obj_name, attr_name = parts
                if obj_name == 'order' and hasattr(order, attr_name):
                    result = result.replace(f'{{{placeholder}}}', str(getattr(order, attr_name)))
                elif obj_name == 'customer' and hasattr(order.customer, attr_name):
                    # Double-check it's not a restricted field
                    if attr_name.lower() not in restricted_fields:
                        result = result.replace(f'{{{placeholder}}}', str(getattr(order.customer, attr_name)))
    
    return result
