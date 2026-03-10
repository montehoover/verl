class Customer:
    def __init__(self, full_name, email_address, home_address, bank_card):
        self.full_name = full_name
        self.email_address = email_address
        self.home_address = home_address
        self.bank_card = bank_card

class Order:
    def __init__(self, order_id, client, item_list, total_amount, payment_method):
        self.order_id = order_id
        self.client = client
        self.item_list = item_list
        self.total_amount = total_amount
        self.payment_method = payment_method

invoices = {
    "INV001": Order("INV001", 
                    Customer("Alice Smith", "alice@domain.com", "789 Pine St, Anytown, USA", "9876-5432-1098-7654"),
                    ["Gadget A", "Gadget B"],
                    199.99,
                    "Credit Card")
}

def compose_invoice_template(invoice_id: str, receipt_format: str) -> str:
    if invoice_id not in invoices:
        raise ValueError(f"Invalid order ID: {invoice_id}")
    
    order = invoices[invoice_id]
    
    # Create a safe namespace for template substitution
    safe_namespace = {
        'order': {
            'order_id': order.order_id,
            'item_list': order.item_list,
            'total_amount': order.total_amount,
            'payment_method': order.payment_method
        },
        'customer': {
            'full_name': order.client.full_name,
            'email_address': order.client.email_address
        }
    }
    
    # Check for protected information in the template
    protected_fields = ['bank_card', 'home_address']
    for field in protected_fields:
        if f'customer.{field}' in receipt_format:
            raise ValueError(f"Access to protected customer information '{field}' is not allowed")
    
    # Replace placeholders in the template
    result = receipt_format
    
    # Replace order placeholders
    for key, value in safe_namespace['order'].items():
        placeholder = f'{{order.{key}}}'
        if placeholder in result:
            result = result.replace(placeholder, str(value))
    
    # Replace customer placeholders
    for key, value in safe_namespace['customer'].items():
        placeholder = f'{{customer.{key}}}'
        if placeholder in result:
            result = result.replace(placeholder, str(value))
    
    return result
