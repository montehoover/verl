import re
from typing import Dict, Any, Set

# Placeholder for the Order class.
# In a real application, this would be a more complex object.
class Order:
    def __init__(self, order_id: str, details: Dict[str, Any]):
        self.order_id = order_id
        self.details = details

    def __repr__(self) -> str:
        return f"Order(order_id='{self.order_id}', details={self.details})"

# Sample orders database (dictionary)
# In a real application, this would come from a database or other persistent storage.
SAMPLE_ORDERS_DB: Dict[str, Order] = {
    "txn_123": Order(order_id="order_abc", details={"item": "Laptop", "quantity": 1, "price": 1200.00}),
    "txn_456": Order(order_id="order_def", details={"item": "Mouse", "quantity": 2, "price": 25.00}),
    "txn_789": Order(order_id="order_ghi", details={"item": "Keyboard", "quantity": 1, "price": 75.00}),
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
    "customer.name",
    "customer.email",
    "customer.address",
    "order.id",
    "order.date",
    "order.total",
    "order.items_list", # e.g., a formatted string of items
    "shop.name",
    "shop.contact_email",
}

SENSITIVE_PLACEHOLDERS: Set[str] = {
    "customer.password",
    "customer.credit_card_number",
    "customer.cvv",
    "order.payment_auth_code",
    "internal.user_id", # Example of an internal, non-customer facing placeholder
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


if __name__ == '__main__':
    # Example Usage:
    print("Available orders:", SAMPLE_ORDERS_DB)

    # Test case 1: Valid transaction ID
    try:
        order1 = get_order_by_id("txn_123", SAMPLE_ORDERS_DB)
        print(f"\nFound order for txn_123: {order1}")
    except ValueError as e:
        print(f"\nError for txn_123: {e}")

    # Test case 2: Another valid transaction ID
    try:
        order2 = get_order_by_id("txn_456", SAMPLE_ORDERS_DB)
        print(f"\nFound order for txn_456: {order2}")
    except ValueError as e:
        print(f"\nError for txn_456: {e}")

    # Test case 3: Invalid transaction ID (not found)
    try:
        order_not_found = get_order_by_id("txn_000", SAMPLE_ORDERS_DB)
        print(f"\nFound order for txn_000: {order_not_found}")
    except ValueError as e:
        print(f"\nError for txn_000: {e}")

    # Test case 4: Invalid transaction ID (empty string)
    try:
        order_empty_id = get_order_by_id("", SAMPLE_ORDERS_DB)
        print(f"\nFound order for empty ID: {order_empty_id}")
    except ValueError as e:
        print(f"\nError for empty ID: {e}")

    # Test case 5: Invalid transaction ID (wrong type)
    try:
        order_wrong_type = get_order_by_id(123, SAMPLE_ORDERS_DB) # type: ignore
        print(f"\nFound order for wrong type ID: {order_wrong_type}")
    except ValueError as e:
        print(f"\nError for wrong type ID: {e}")

    print("\n--- Testing Template Placeholder Validation ---")

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
