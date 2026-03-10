import re

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

# Predefined dictionary of invoices
invoices = {
    "INV001": Order("INV001", 
                    Customer("Alice Smith", "alice@domain.com", "789 Pine St, Anytown, USA", "9876-5432-1098-7654"),
                    ["Gadget A", "Gadget B"],
                    199.99,
                    "Credit Card")
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

def compose_invoice_template(invoice_id, receipt_format):
    # Check if order exists
    if invoice_id not in invoices:
        raise ValueError(f"Order ID '{invoice_id}' not found")
    
    order = invoices[invoice_id]
    
    # Check for protected information access attempts
    protected_patterns = [
        r'\{.*bank_card.*\}',
        r'\{.*credit.*card.*\}',
        r'\{.*card.*number.*\}',
        r'\{.*ssn.*\}',
        r'\{.*social.*security.*\}',
        r'\{.*password.*\}'
    ]
    
    for pattern in protected_patterns:
        if re.search(pattern, receipt_format, re.IGNORECASE):
            raise ValueError("Attempt to access protected customer information")
    
    # Replace placeholders
    formatted_receipt = receipt_format
    
    # Customer placeholders
    formatted_receipt = formatted_receipt.replace('{customer.full_name}', order.client.full_name)
    formatted_receipt = formatted_receipt.replace('{customer.email}', order.client.email_address)
    formatted_receipt = formatted_receipt.replace('{customer.email_address}', order.client.email_address)
    formatted_receipt = formatted_receipt.replace('{customer.address}', order.client.home_address)
    formatted_receipt = formatted_receipt.replace('{customer.home_address}', order.client.home_address)
    
    # Order placeholders
    formatted_receipt = formatted_receipt.replace('{order.order_id}', order.order_id)
    formatted_receipt = formatted_receipt.replace('{order.total_amount}', str(order.total_amount))
    formatted_receipt = formatted_receipt.replace('{order.items}', ', '.join(order.item_list))
    formatted_receipt = formatted_receipt.replace('{order.payment_method}', order.payment_method)
    
    # Client alias (in case template uses 'client' instead of 'customer')
    formatted_receipt = formatted_receipt.replace('{client.full_name}', order.client.full_name)
    formatted_receipt = formatted_receipt.replace('{client.email_address}', order.client.email_address)
    formatted_receipt = formatted_receipt.replace('{client.home_address}', order.client.home_address)
    
    return formatted_receipt
