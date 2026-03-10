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

def compose_invoice_template(invoice_id, receipt_format):
    """Generate a receipt string by replacing placeholders with actual order data."""
    if invoice_id not in invoices:
        raise ValueError(f"Invalid order ID: {invoice_id}")
    
    order = invoices[invoice_id]
    
    # Protected customer information that should not be accessible
    protected_fields = ['bank_card', 'home_address']
    
    # Check for protected field access attempts
    for field in protected_fields:
        if f'{{customer.{field}}}' in receipt_format:
            raise ValueError(f"Access to protected customer information '{field}' is not allowed")
    
    # Create replacement dictionary
    replacements = {
        '{order.order_id}': order.order_id,
        '{order.total_amount}': str(order.total_amount),
        '{order.payment_method}': order.payment_method,
        '{order.item_list}': ', '.join(order.item_list),
        '{customer.full_name}': order.client.full_name,
        '{customer.email_address}': order.client.email_address
    }
    
    # Replace all placeholders in the receipt format
    result = receipt_format
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    
    return result
