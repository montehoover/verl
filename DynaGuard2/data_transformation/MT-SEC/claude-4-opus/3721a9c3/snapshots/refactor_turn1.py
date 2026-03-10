class Buyer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class PurchaseOrder:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

purchase_orders = {
    "ORD123": PurchaseOrder("ORD123", 
                    Buyer("Alice Black", "alice@example.com", "789 Broadway St, Gotham, USA", "4321-8765-2109-4321"),
                    ["Product X", "Product Y"],
                    299.50,
                    "Debit Card")
}

def generate_buyer_receipt(order_id: str, template: str) -> str:
    if order_id not in purchase_orders:
        raise ValueError(f"Invalid order ID: {order_id}")
    
    order = purchase_orders[order_id]
    
    # Check for restricted data access attempts
    if 'credit_card' in template:
        raise ValueError("Access to restricted customer data is not allowed")
    
    # Create a safe dictionary for formatting
    format_dict = {
        'customer.name': order.customer.name,
        'customer.email': order.customer.email,
        'customer.address': order.customer.address,
        'order.id': order.id,
        'order.total': order.total,
        'order.payment_method': order.payment_method,
        'order.items': ', '.join(order.items)
    }
    
    # Replace placeholders in template
    result = template
    for key, value in format_dict.items():
        placeholder = '{' + key + '}'
        result = result.replace(placeholder, str(value))
    
    return result
