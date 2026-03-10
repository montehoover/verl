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
    # Check if the order ID is valid
    if invoice_id not in invoices:
        raise ValueError(f"Invalid order ID: {invoice_id}")
    
    # Get the order object
    order = invoices[invoice_id]
    customer = order.client
    
    # Check for protected information in the template
    protected_fields = ['bank_card']
    for field in protected_fields:
        if f'customer.{field}' in receipt_format or f'client.{field}' in receipt_format:
            raise ValueError(f"Access to protected customer information '{field}' is not allowed")
    
    # Create a dictionary with all the replaceable values
    replacements = {
        'order.order_id': order.order_id,
        'order.item_list': ', '.join(order.item_list),
        'order.total_amount': order.total_amount,
        'order.payment_method': order.payment_method,
        'customer.full_name': customer.full_name,
        'customer.email_address': customer.email_address,
        'customer.home_address': customer.home_address,
        'client.full_name': customer.full_name,
        'client.email_address': customer.email_address,
        'client.home_address': customer.home_address
    }
    
    # Replace all placeholders in the receipt format
    result = receipt_format
    for key, value in replacements.items():
        placeholder = f'{{{key}}}'
        result = result.replace(placeholder, str(value))
    
    return result
