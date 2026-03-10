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

def create_receipt_for_transaction(transaction_id, template_format):
    if transaction_id not in orders:
        raise ValueError(f"Transaction {transaction_id} not found in our system.")
    
    order = orders[transaction_id]
    
    # Check for protected customer data
    protected_fields = ['credit_card', 'customer.credit_card']
    for field in protected_fields:
        if f'{{{field}}}' in template_format:
            raise ValueError(f"Access to protected customer data '{field}' is not allowed.")
    
    # Create replacement dictionary
    replacements = {
        'order.id': order.id,
        'order.total': order.total,
        'order.payment_method': order.payment_method,
        'order.items': ', '.join(order.items),
        'customer.name': order.customer.name,
        'customer.email': order.customer.email,
        'customer.address': order.customer.address
    }
    
    # Replace placeholders in template
    result = template_format
    for key, value in replacements.items():
        result = result.replace(f'{{{key}}}', str(value))
    
    return result
