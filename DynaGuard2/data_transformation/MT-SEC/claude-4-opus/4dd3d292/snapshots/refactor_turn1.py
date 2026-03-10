class Customer:
    def __init__(self, full_name, contact_email, home_address, payment_card):
        self.full_name = full_name
        self.contact_email = contact_email
        self.home_address = home_address
        self.payment_card = payment_card

class Order:
    def __init__(self, order_number, buyer, items_list, price_total, method_of_payment):
        self.order_number = order_number
        self.buyer = buyer
        self.items_list = items_list
        self.price_total = price_total
        self.method_of_payment = method_of_payment

purchases = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item A", "Item B"],
                    99.99,
                    "Credit Card")
}

def create_purchase_receipt(order_id: str, receipt_template: str) -> str:
    if order_id not in purchases:
        raise ValueError(f"Invalid order ID: {order_id}")
    
    order = purchases[order_id]
    
    # Check for protected information access attempts
    protected_fields = ['payment_card', 'home_address']
    for field in protected_fields:
        if f'customer.{field}' in receipt_template or f'buyer.{field}' in receipt_template:
            raise ValueError(f"Access to protected customer information '{field}' is not allowed")
    
    # Create a safe customer object with protected fields removed
    safe_customer = {
        'full_name': order.buyer.full_name,
        'contact_email': order.buyer.contact_email
    }
    
    # Create context for template replacement
    context = {
        'order': {
            'order_number': order.order_number,
            'items_list': order.items_list,
            'price_total': order.price_total,
            'method_of_payment': order.method_of_payment
        },
        'customer': safe_customer,
        'buyer': safe_customer  # Support both 'customer' and 'buyer' prefixes
    }
    
    # Replace placeholders in template
    result = receipt_template
    
    # Replace order fields
    for field, value in context['order'].items():
        placeholder = f'{{order.{field}}}'
        if placeholder in result:
            if isinstance(value, list):
                value = ', '.join(value)
            result = result.replace(placeholder, str(value))
    
    # Replace customer/buyer fields
    for prefix in ['customer', 'buyer']:
        for field, value in context[prefix].items():
            placeholder = f'{{{prefix}.{field}}}'
            result = result.replace(placeholder, str(value))
    
    return result
