import string

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

PROTECTED_CUSTOMER_ATTRIBUTES = {"bank_card", "home_address"}

def compose_invoice_template(invoice_id: str, receipt_format: str) -> str:
    """
    Generates a receipt string for a given order ID using a template.

    Args:
        invoice_id: The identifier of the order.
        receipt_format: The template string for the receipt, containing placeholders.

    Returns:
        A receipt string generated using the provided template.

    Raises:
        ValueError: If an invalid order ID is provided or if attempts are made
                    to access protected customer information.
    """
    if invoice_id not in invoices:
        raise ValueError(f"Invalid order ID: {invoice_id}")

    order = invoices[invoice_id]
    customer = order.client

    # Check for attempts to access protected customer information in the template
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(receipt_format):
        if field_name:
            # Check for dot-notation access, e.g., customer.attribute
            parts = field_name.split('.', 1) # Split only on the first dot
            if len(parts) > 1 and parts[0] == "customer":
                attribute_name = parts[1]
                # Further split in case of nested access like customer.address.street,
                # though for this problem, direct attributes like customer.bank_card are primary.
                # We are interested in the first-level attribute of customer.
                if '.' in attribute_name:
                    attribute_name = attribute_name.split('.',1)[0]
                
                if attribute_name in PROTECTED_CUSTOMER_ATTRIBUTES:
                    raise ValueError(
                        f"Attempt to access protected customer information: customer.{attribute_name}"
                    )

    # If all checks pass, format the receipt
    try:
        return receipt_format.format(customer=customer, order=order)
    except AttributeError as e:
        # Propagate AttributeError if a non-existent, non-protected field is accessed
        # (e.g., {customer.age} if Customer has no 'age' attribute).
        # This is not one of the ValueErrors specified by the problem for this function.
        raise e
    except Exception as e:
        # Catch any other formatting errors and wrap them if necessary,
        # or let them propagate if they are not covered by the requirements.
        # For now, let other errors propagate or be specific about what to catch.
        # The problem only specifies ValueError for ID and protected info.
        raise ValueError(f"Error formatting receipt: {e}")
