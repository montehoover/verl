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

def get_order_details(order_id, template_string=None):
    if order_id not in orders:
        raise ValueError(f"Order ID '{order_id}' does not exist")
    
    order = orders[order_id]
    
    if template_string is None:
        print(f"Order ID: {order.id}")
        print(f"Customer Name: {order.customer.name}")
        print(f"Total Amount: ${order.total:.2f}")
        return
    
    # Create a dictionary of available placeholders
    placeholders = {
        'order_id': order.id,
        'customer_name': order.customer.name,
        'total': f"${order.total:.2f}",
        'payment_method': order.payment_method,
        'num_items': len(order.items)
    }
    
    # Replace placeholders in the template string
    formatted_string = template_string
    for key, value in placeholders.items():
        placeholder = f'{{{key}}}'
        if placeholder in formatted_string:
            formatted_string = formatted_string.replace(placeholder, str(value))
    
    return formatted_string
