import re
from typing import Dict, Any, Set

# --- Data Models ---
class Customer:
    def __init__(self, name: str, email: str, address: str, credit_card: str):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card  # Sensitive data

    def __repr__(self) -> str:
        # Mask credit card in representation for security
        return f"Customer(name='{self.name}', email='{self.email}', address='{self.address}', credit_card='****')"

class Order:
    def __init__(self, id: str, customer: Customer, items: list[str], total: float, payment_method: str):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

    def __repr__(self) -> str:
        return f"Order(id='{self.id}', customer={self.customer}, items={self.items}, total={self.total}, payment_method='{self.payment_method}')"

# Orders database
ORDERS_DATA: Dict[str, Order] = {
    "ORD001": Order(
        id="ORD001",
        customer=Customer(
            name="Jane Doe",
            email="jane@example.com",
            address="123 Main St, Anytown, USA",
            credit_card="1234-5678-9012-3456" # This is sensitive
        ),
        items=["Wireless Mouse", "USB-C Hub"],
        total=74.98,
        payment_method="Credit Card"
    ),
    "ORD002": Order(
        id="ORD002",
        customer=Customer(
            name="John Smith",
            email="john.smith@example.com",
            address="456 Oak Ave, Otherville, USA",
            credit_card="9876-5432-1098-7654" # This is sensitive
        ),
        items=["Mechanical Keyboard", "Gaming Headset", "Mouse Pad"],
        total=215.50,
        payment_method="PayPal"
    )
}

def get_order_by_id(transaction_id: str, orders_db: Dict[str, Order]) -> Order:
    """
    Retrieves an Order object from a dictionary of orders using the transaction ID.

    Args:
        transaction_id: The ID of the transaction to retrieve.
        orders_db: A dictionary where keys are transaction IDs and values are Order objects.

    Returns:
        The Order object corresponding to the transaction ID.

    Raises:
        ValueError: If the transaction ID is invalid (e.g., not a string, empty)
                    or not found in the orders_db.
    """
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ValueError("Transaction ID must be a non-empty string.")

    order = orders_db.get(transaction_id)
    if order is None:
        raise ValueError(f"Order with transaction ID '{transaction_id}' not found.")
    
    return order

# --- Template Placeholder Validation ---
ALLOWED_PLACEHOLDERS: Set[str] = {
    "customer.name",        # Corresponds to order.customer.name
    "customer.email",       # Corresponds to order.customer.email
    "customer.address",     # Corresponds to order.customer.address
    "order.id",             # Corresponds to order.id
    "order.total",          # Corresponds to order.total
    "order.items_list",     # Corresponds to ", ".join(order.items)
    "order.payment_method", # Corresponds to order.payment_method
}

SENSITIVE_PLACEHOLDERS: Set[str] = {
    "customer.password",            # Generic sensitive placeholder
    "customer.credit_card_number",  # Corresponds to order.customer.credit_card
    "customer.cvv",                 # Generic sensitive placeholder
    "order.payment_auth_code",      # Generic sensitive placeholder
    "internal.user_id",             # Example of an internal, non-customer facing placeholder
}

def check_template_placeholders(template_string: str) -> bool:
    """
    Checks a template string for valid and non-sensitive placeholders.

    Placeholders are expected in the format {object.attribute}.

    Args:
        template_string: The template string to check.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If an invalid or sensitive placeholder is found.
    """
    if not isinstance(template_string, str):
        raise ValueError("Template string must be a string.")

    # Regex to find all occurrences of {placeholder}
    found_placeholders = re.findall(r'\{([^}]+)\}', template_string)

    if not found_placeholders:
        return True # No placeholders to check

    for placeholder in found_placeholders:
        if placeholder in SENSITIVE_PLACEHOLDERS:
            raise ValueError(f"Sensitive placeholder '{placeholder}' found in template.")
        if placeholder not in ALLOWED_PLACEHOLDERS:
            raise ValueError(f"Invalid placeholder '{placeholder}' found in template.")
            
    return True

# --- Receipt Generation ---
def create_receipt_for_transaction(transaction_id: str, template_format: str, all_orders: Dict[str, Order]) -> str:
    """
    Creates a formatted receipt string for a given transaction ID using a template.

    Args:
        transaction_id: The ID of the transaction (maps to Order ID).
        template_format: The string template for the receipt.
        all_orders: A dictionary of all orders, keyed by transaction/order ID.

    Returns:
        A string representing the formatted receipt.

    Raises:
        ValueError: If the transaction ID is not found, or if the template
                    contains invalid or sensitive placeholders.
    """
    # 1. Validate template placeholders first
    # This will raise ValueError if invalid/sensitive placeholders are found
    check_template_placeholders(template_format)

    # 2. Retrieve the order
    # This will raise ValueError if transaction_id is invalid or not found
    order = get_order_by_id(transaction_id, all_orders)

    # 3. Substitute placeholders with actual data
    # At this point, we know all placeholders in template_format are allowed
    # and not sensitive.
    receipt_string = template_format

    # Perform substitutions based on ALLOWED_PLACEHOLDERS
    # Note: Direct string replacement is safe here because check_template_placeholders
    # has already validated the placeholders.
    if "{customer.name}" in receipt_string:
        receipt_string = receipt_string.replace("{customer.name}", order.customer.name)
    if "{customer.email}" in receipt_string:
        receipt_string = receipt_string.replace("{customer.email}", order.customer.email)
    if "{customer.address}" in receipt_string:
        receipt_string = receipt_string.replace("{customer.address}", order.customer.address)
    if "{order.id}" in receipt_string:
        receipt_string = receipt_string.replace("{order.id}", order.id)
    if "{order.total}" in receipt_string:
        receipt_string = receipt_string.replace("{order.total}", str(order.total))
    if "{order.items_list}" in receipt_string:
        receipt_string = receipt_string.replace("{order.items_list}", ", ".join(order.items))
    if "{order.payment_method}" in receipt_string:
        receipt_string = receipt_string.replace("{order.payment_method}", order.payment_method)
    
    return receipt_string


if __name__ == '__main__':
    print("Available orders:", ORDERS_DATA)

    print("\n--- Testing get_order_by_id ---")
    # Test case 1: Valid transaction ID
    try:
        order1 = get_order_by_id("ORD001", ORDERS_DATA)
        print(f"\nFound order for ORD001: {order1}")
    except ValueError as e:
        print(f"\nError for ORD001: {e}")

    # Test case 2: Another valid transaction ID
    try:
        order2 = get_order_by_id("ORD002", ORDERS_DATA)
        print(f"\nFound order for ORD002: {order2}")
    except ValueError as e:
        print(f"\nError for ORD002: {e}")

    # Test case 3: Invalid transaction ID (not found)
    try:
        order_not_found = get_order_by_id("ORD999", ORDERS_DATA)
        print(f"\nFound order for ORD999: {order_not_found}")
    except ValueError as e:
        print(f"\nError for ORD999: {e}")

    # Test case 4: Invalid transaction ID (empty string)
    try:
        order_empty_id = get_order_by_id("", ORDERS_DATA)
        print(f"\nFound order for empty ID: {order_empty_id}")
    except ValueError as e:
        print(f"\nError for empty ID: {e}")

    # Test case 5: Invalid transaction ID (wrong type)
    try:
        order_wrong_type = get_order_by_id(123, ORDERS_DATA) # type: ignore
        print(f"\nFound order for wrong type ID: {order_wrong_type}")
    except ValueError as e:
        print(f"\nError for wrong type ID: {e}")

    print("\n--- Testing check_template_placeholders (no changes to tests needed here unless ALLOWED/SENSITIVE changed significantly) ---")

    # Test case 1: Valid template
    valid_template = "Hello {customer.name}, your order {order.id} for ${order.total} is confirmed. Shop: {shop.name}."
    try:
        check_template_placeholders(valid_template)
        print(f"\nTemplate 1 (Valid): '{valid_template}' - Check PASSED")
    except ValueError as e:
        print(f"\nTemplate 1 (Valid): '{valid_template}' - Check FAILED: {e}")

    # Test case 2: Template with an invalid placeholder
    invalid_template_1 = "Dear {customer.name}, your order {order.status} is processing." # order.status is not allowed
    try:
        check_template_placeholders(invalid_template_1)
        print(f"\nTemplate 2 (Invalid): '{invalid_template_1}' - Check PASSED (Error expected)")
    except ValueError as e:
        print(f"\nTemplate 2 (Invalid): '{invalid_template_1}' - Check FAILED as expected: {e}")

    # Test case 3: Template with a sensitive placeholder
    sensitive_template = "User {customer.name}, your password hint is {customer.password}." # customer.password is sensitive
    try:
        check_template_placeholders(sensitive_template)
        print(f"\nTemplate 3 (Sensitive): '{sensitive_template}' - Check PASSED (Error expected)")
    except ValueError as e:
        print(f"\nTemplate 3 (Sensitive): '{sensitive_template}' - Check FAILED as expected: {e}")

    # Test case 4: Template with mixed valid, invalid, and sensitive placeholders
    mixed_template = "Hi {customer.name}, your order {order.id}. Don't share {customer.credit_card_number}. Unknown: {order.details}"
    try:
        check_template_placeholders(mixed_template)
        print(f"\nTemplate 4 (Mixed): '{mixed_template}' - Check PASSED (Error expected)")
    except ValueError as e:
        print(f"\nTemplate 4 (Mixed): '{mixed_template}' - Check FAILED as expected: {e}")
        # Note: The function will raise error on the first invalid/sensitive placeholder it encounters.

    # Test case 5: Template with no placeholders
    no_placeholder_template = "This is a static message."
    try:
        check_template_placeholders(no_placeholder_template)
        print(f"\nTemplate 5 (No placeholders): '{no_placeholder_template}' - Check PASSED")
    except ValueError as e:
        print(f"\nTemplate 5 (No placeholders): '{no_placeholder_template}' - Check FAILED: {e}")

    # Test case 6: Empty template string
    empty_template = ""
    try:
        check_template_placeholders(empty_template)
        print(f"\nTemplate 6 (Empty): '{empty_template}' - Check PASSED")
    except ValueError as e:
        print(f"\nTemplate 6 (Empty): '{empty_template}' - Check FAILED: {e}")
    
    # Test case 7: Template with only sensitive placeholder
    only_sensitive_template = "Your secret: {internal.user_id}"
    try:
        check_template_placeholders(only_sensitive_template)
        print(f"\nTemplate 7 (Only Sensitive): '{only_sensitive_template}' - Check PASSED (Error expected)")
    except ValueError as e:
        print(f"\nTemplate 7 (Only Sensitive): '{only_sensitive_template}' - Check FAILED as expected: {e}")

    # Test case 8: Template with only invalid placeholder
    only_invalid_template = "Data: {nonexistent.field}"
    try:
        check_template_placeholders(only_invalid_template)
        print(f"\nTemplate 8 (Only Invalid): '{only_invalid_template}' - Check PASSED (Error expected)")
    except ValueError as e:
        print(f"\nTemplate 8 (Only Invalid): '{only_invalid_template}' - Check FAILED as expected: {e}")

    print("\n--- Testing create_receipt_for_transaction ---")

    # Test case 1: Successful receipt generation
    valid_receipt_template = "Receipt for Order {order.id}:\nCustomer: {customer.name} ({customer.email})\nAddress: {customer.address}\nItems: {order.items_list}\nTotal: ${order.total}\nPaid via: {order.payment_method}"
    try:
        receipt1 = create_receipt_for_transaction("ORD001", valid_receipt_template, ORDERS_DATA)
        print(f"\nGenerated Receipt for ORD001:\n{receipt1}")
    except ValueError as e:
        print(f"\nError generating receipt for ORD001: {e}")

    # Test case 2: Transaction ID not found
    try:
        receipt_not_found = create_receipt_for_transaction("ORD999", valid_receipt_template, ORDERS_DATA)
        print(f"\nGenerated Receipt for ORD999: {receipt_not_found}")
    except ValueError as e:
        print(f"\nError generating receipt for ORD999 (expected): {e}")

    # Test case 3: Template with invalid placeholder (e.g., {order.status})
    invalid_placeholder_template = "Order {order.id} status: {order.status}" # {order.status} is not in ALLOWED_PLACEHOLDERS
    try:
        receipt_invalid_placeholder = create_receipt_for_transaction("ORD001", invalid_placeholder_template, ORDERS_DATA)
        print(f"\nGenerated Receipt with invalid placeholder: {receipt_invalid_placeholder}")
    except ValueError as e:
        print(f"\nError with invalid placeholder template (expected): {e}")

    # Test case 4: Template with sensitive placeholder (e.g., {customer.credit_card_number})
    sensitive_placeholder_template = "Dear {customer.name}, your card {customer.credit_card_number} was charged."
    try:
        receipt_sensitive_placeholder = create_receipt_for_transaction("ORD001", sensitive_placeholder_template, ORDERS_DATA)
        print(f"\nGenerated Receipt with sensitive placeholder: {receipt_sensitive_placeholder}")
    except ValueError as e:
        print(f"\nError with sensitive placeholder template (expected): {e}")
    
    # Test case 5: Template with a placeholder that is allowed but not present in this specific template string
    # (e.g. {customer.address} is allowed, but not used in this template)
    # This should still work fine.
    simple_template = "Order {order.id} for {customer.name}. Total: ${order.total}."
    try:
        receipt_simple = create_receipt_for_transaction("ORD002", simple_template, ORDERS_DATA)
        print(f"\nGenerated Simple Receipt for ORD002:\n{receipt_simple}")
    except ValueError as e:
        print(f"\nError generating simple receipt for ORD002: {e}")
