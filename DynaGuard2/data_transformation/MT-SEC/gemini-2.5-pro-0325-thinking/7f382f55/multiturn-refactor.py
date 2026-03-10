import string
import logging
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

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define fields exposed through the template's {customer.*} and {order.*} placeholders.
EXPOSED_CUSTOMER_FIELDS = {"name", "email", "address"}
EXPOSED_ORDER_FIELDS = {"id", "items", "total", "payment_method"}

def _get_order_details(invoice_id: str) -> Order:
    """Retrieves order details for the given invoice ID."""
    if invoice_id not in invoices:
        raise ValueError(f"Invalid order ID: {invoice_id}")
    return invoices[invoice_id]

def _validate_receipt_template(receipt_format: str):
    """
    Validates the placeholders in the receipt format string.
    Uses module-level EXPOSED_CUSTOMER_FIELDS and EXPOSED_ORDER_FIELDS.
    Raises ValueError for invalid placeholders or access to protected/non-exposed fields.
    """
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
                raise ValueError(f"Invalid object type '{obj_name}' in placeholder: '{field_name}'. Must be 'customer' or 'order'.")

def _create_safe_views(order: Order) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Creates safe, read-only views for customer and order data for template formatting."""
    customer = order.client
    
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
    return safe_customer_view, safe_order_view

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
    logger.info(f"Starting receipt generation for invoice ID: {invoice_id}")
    logger.debug(f"Using receipt template: {receipt_format}")

    try:
        order = _get_order_details(invoice_id)
        _validate_receipt_template(receipt_format)  # Uses module-level constants for exposed fields
        
        safe_customer_view, safe_order_view = _create_safe_views(order)

        # Perform the formatting using the safe views
        # str.format() itself can raise ValueError for issues like bad format specifiers (e.g. {order.total:xyz})
        generated_receipt = receipt_format.format(customer=safe_customer_view, order=safe_order_view)
        
        logger.info(f"Successfully generated receipt for invoice ID: {invoice_id}")
        logger.debug(f"Generated receipt content: {generated_receipt}")
        return generated_receipt
    except ValueError as e:
        logger.error(f"Error generating receipt for invoice ID {invoice_id}: {e}")
        raise # Re-raise the exception after logging
    except Exception as e:
        logger.error(f"An unexpected error occurred while generating receipt for invoice ID {invoice_id}: {e}", exc_info=True)
        raise # Re-raise the exception after logging
