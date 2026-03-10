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

def get_order_by_id(invoice_id: str, order_database: dict) -> Order:
    """Retrieve an order from the database by its ID."""
    if invoice_id not in order_database:
        raise ValueError(f"Invalid order ID: {invoice_id}")
    return order_database[invoice_id]

def validate_template_security(template: str, protected_fields: list[str]) -> None:
    """Validate that the template doesn't contain protected fields."""
    for field in protected_fields:
        if f'customer.{field}' in template:
            raise ValueError(f"Access to protected customer information '{field}' is not allowed")

def create_safe_data_namespace(order: Order) -> dict:
    """Create a safe namespace with allowed order and customer data."""
    return {
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

def format_receipt_template(template: str, data_namespace: dict) -> str:
    """Replace placeholders in the template with actual data."""
    result = template
    
    # Replace order placeholders
    for key, value in data_namespace['order'].items():
        placeholder = f'{{order.{key}}}'
        if placeholder in result:
            result = result.replace(placeholder, str(value))
    
    # Replace customer placeholders
    for key, value in data_namespace['customer'].items():
        placeholder = f'{{customer.{key}}}'
        if placeholder in result:
            result = result.replace(placeholder, str(value))
    
    return result

def compose_invoice_template(invoice_id: str, receipt_format: str) -> str:
    """Generate a receipt for an order using a template format."""
    # Retrieve the order
    order = get_order_by_id(invoice_id, invoices)
    
    # Validate template security
    protected_fields = ['bank_card', 'home_address']
    validate_template_security(receipt_format, protected_fields)
    
    # Create safe data namespace
    safe_namespace = create_safe_data_namespace(order)
    
    # Format the receipt
    return format_receipt_template(receipt_format, safe_namespace)
