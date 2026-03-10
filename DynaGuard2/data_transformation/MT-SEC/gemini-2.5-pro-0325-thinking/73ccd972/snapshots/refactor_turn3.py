import re
import logging
import os

# Setup logging
log_file_path = os.path.join(os.getcwd(), "receipt_generation.log")
logger = logging.getLogger("receipt_logger")
logger.setLevel(logging.INFO)
# Prevent duplicate handlers if this script is run multiple times in the same session (e.g. in a notebook)
if not logger.handlers:
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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

# Allowed attributes for placeholders to prevent arbitrary code execution via getattr
ALLOWED_CUSTOMER_ATTRIBUTES = {"name", "email", "address"}
ALLOWED_ORDER_ATTRIBUTES = {"id", "items", "total", "payment_method"}
RESTRICTED_CUSTOMER_ATTRIBUTES = {"credit_card"}

# Regex to find placeholders like {customer.name} or {order.total}
# It ensures that object_name and attribute_name are valid identifiers
PLACEHOLDER_REGEX = r"{([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)}"


def _get_placeholder_value(object_name: str, attribute_name: str, customer_obj: Customer, order_obj: Order) -> str:
    """
    Retrieves and validates a placeholder's value.
    Raises ValueError for invalid/restricted attributes or object types.
    """
    if object_name == "customer":
        if attribute_name in RESTRICTED_CUSTOMER_ATTRIBUTES:
            error_msg = f"Access to restricted customer data ({attribute_name}) is not allowed."
            logger.error(f"Placeholder error for customer: {error_msg}")
            raise ValueError(error_msg)
        if attribute_name not in ALLOWED_CUSTOMER_ATTRIBUTES:
            error_msg = f"Invalid or disallowed customer attribute: {attribute_name}"
            logger.error(f"Placeholder error for customer: {error_msg}")
            raise ValueError(error_msg)
        value = getattr(customer_obj, attribute_name)
    elif object_name == "order":
        if attribute_name not in ALLOWED_ORDER_ATTRIBUTES:
            error_msg = f"Invalid or disallowed order attribute: {attribute_name}"
            logger.error(f"Placeholder error for order: {error_msg}")
            raise ValueError(error_msg)
        value = getattr(order_obj, attribute_name)
    else:
        error_msg = f"Invalid object type in placeholder: {object_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return str(value)


def _process_template(receipt_template: str, customer_obj: Customer, order_obj: Order) -> str:
    """
    Processes the receipt template, replacing placeholders with actual data.
    """
    def replace_match(match):
        object_name = match.group(1)
        attribute_name = match.group(2)
        return _get_placeholder_value(object_name, attribute_name, customer_obj, order_obj)

    try:
        formatted_receipt = re.sub(PLACEHOLDER_REGEX, replace_match, receipt_template)
    except ValueError as ve: # Catch ValueErrors raised from _get_placeholder_value
        logger.error(f"Error during placeholder replacement: {ve}")
        raise
    except Exception as e:
        # Catch other potential errors during getattr or regex processing
        logger.error(f"Unexpected error processing receipt template: {e}")
        raise ValueError(f"Error processing receipt template: {e}")

    # Check if any placeholders remain, which might indicate an invalid format not caught by regex
    # or an attribute that was not processed.
    if re.search(r"{.*}", formatted_receipt): # A simple check for any remaining curly braces
        # This check might be refined if specific unreplaced placeholder formats need to be identified.
        # For now, it assumes that if _get_placeholder_value didn't handle it and it's still there,
        # it might be an issue not covered by PLACEHOLDER_REGEX or a malformed placeholder.
        # However, with the current strict PLACEHOLDER_REGEX, this is less likely for valid formats.
        pass

    return formatted_receipt


def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    """
    Generates a receipt for a completed order by substituting placeholders in a template.

    Args:
        purchase_id: The unique identifier of the order.
        receipt_template: The template string for generating the receipt.
                          Placeholders are in the format {object.attribute},
                          e.g., {customer.name} or {order.total}.

    Returns:
        The formatted receipt string with placeholders replaced by order and customer details.

    Raises:
        ValueError: If the order ID does not exist, or if a placeholder requests
                    restricted or protected data from the customer.
    """
    logger.info(f"Starting receipt generation for order ID: {purchase_id}. Template: \"{receipt_template[:50]}...\"")
    if purchase_id not in orders:
        error_msg = f"Order ID '{purchase_id}' does not exist."
        logger.error(error_msg)
        raise ValueError(error_msg)

    order_obj = orders[purchase_id]
    customer_obj = order_obj.customer

    try:
        receipt = _process_template(receipt_template, customer_obj, order_obj)
        logger.info(f"Successfully generated receipt for order ID: {purchase_id}")
        return receipt
    except ValueError as e:
        # Errors from _process_template (including those from _get_placeholder_value) are already logged.
        # Re-raise to maintain function contract.
        logger.error(f"Failed to generate receipt for order ID {purchase_id} due to: {e}")
        raise
