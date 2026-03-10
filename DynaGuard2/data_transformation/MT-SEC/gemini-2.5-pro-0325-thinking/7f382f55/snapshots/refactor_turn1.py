import string
from types import SimpleNamespace

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
    """
    Generates a receipt string for a given order ID using a template.

    Args:
        invoice_id: The identifier of the order.
        receipt_format: The template string for the receipt, containing placeholders
                        like '{customer.name}' or '{order.total}'.

    Returns:
        A receipt string generated using the provided template.

    Raises:
        ValueError: If an invalid order ID is provided, or if the template
                    attempts to access protected customer information or invalid fields.
    """
    if invoice_id not in invoices:
        raise ValueError(f"Invalid order ID: {invoice_id}")

    order = invoices[invoice_id]
    customer = order.client

    # Define fields exposed through the template's {customer.*} and {order.*} placeholders.
    # These correspond to the attributes of safe_customer_view and safe_order_view.
    EXPOSED_CUSTOMER_FIELDS = {"name", "email", "address"}
    EXPOSED_ORDER_FIELDS = {"id", "items", "total", "payment_method"}

    # Validate placeholders in the receipt_format string
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(receipt_format):
        if field_name:  # Only process actual field names, ignore literal text
            parts = field_name.split('.', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid placeholder format: '{field_name}'. Expected 'object.attribute'.")
            
            obj_name, attr_name = parts
            
            if obj_name == "customer":
                # Explicitly check for 'bank_card' as a protected field
                if attr_name == "bank_card": 
                    raise ValueError(f"Attempt to access protected customer information: customer.bank_card")
                # Check if the attribute is in the set of allowed exposed fields
                if attr_name not in EXPOSED_CUSTOMER_FIELDS:
                    raise ValueError(f"Attempt to access invalid or non-exposed customer attribute: {field_name}")
            elif obj_name == "order":
                if attr_name not in EXPOSED_ORDER_FIELDS:
                    raise ValueError(f"Attempt to access invalid or non-exposed order attribute: {field_name}")
            else:
                # If the object part of the placeholder is not 'customer' or 'order'
                raise ValueError(f"Invalid object type '{obj_name}' in placeholder: '{field_name}'. Must be 'customer' or 'order'.")

    # Create safe views of customer and order data for formatting.
    # These SimpleNamespace objects will only expose the allowed fields.
    safe_customer_view = SimpleNamespace(
        name=customer.full_name,
        email=customer.email_address,
        address=customer.home_address
    )
    
    safe_order_view = SimpleNamespace(
        id=order.order_id,
        items=", ".join(order.item_list), # Convert item list to a comma-separated string
        total=order.total_amount,
        payment_method=order.payment_method
    )

    # Perform the formatting using the safe views
    # str.format() itself can raise ValueError for issues like bad format specifiers (e.g. {order.total:xyz})
    return receipt_format.format(customer=safe_customer_view, order=safe_order_view)
