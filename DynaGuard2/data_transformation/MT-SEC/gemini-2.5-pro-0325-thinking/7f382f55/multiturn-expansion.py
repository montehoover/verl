import re
from typing import Dict, Set, Any

# Setup code provided by the user
class Customer:
    def __init__(self, full_name: str, email_address: str, home_address: str, bank_card: str):
        self.full_name = full_name
        self.email_address = email_address
        self.home_address = home_address
        self.bank_card = bank_card  # Protected information

class Order:
    def __init__(self, order_id: str, client: Customer, item_list: list[str], total_amount: float, payment_method: str):
        self.order_id = order_id
        self.client = client
        self.item_list = item_list
        self.total_amount = total_amount
        self.payment_method = payment_method

invoices: Dict[str, Order] = {
    "INV001": Order("INV001",
                    Customer("Alice Smith", "alice@domain.com", "789 Pine St, Anytown, USA", "9876-5432-1098-7654"),
                    ["Gadget A", "Gadget B"],
                    199.99,
                    "Credit Card"),
    "INV002": Order("INV002",
                    Customer("Bob Johnson", "bob@work.net", "123 Oak Ln, Sometown, USA", "1234-5678-9012-3456"),
                    ["Software Suite", "USB Drive (32GB)"],
                    75.50,
                    "PayPal")
}

def get_order_by_id(order_id: str) -> Order:
    """
    Retrieves an order by its ID from the predefined INVOICES.

    Args:
        order_id: The ID of the order to retrieve.

    Returns:
        The Order object if found.

    Raises:
        ValueError: If the order ID is not found in invoices.
    """
    order = invoices.get(order_id)
    if order is None:
        raise ValueError(f"Order with ID '{order_id}' not found.")
    return order

# Define valid placeholders
# This maps an object type (e.g., "customer", "order") to a set of its valid attributes.
VALID_PLACEHOLDERS: Dict[str, Set[str]] = {
    "order": {"order_id", "item_list", "total_amount", "payment_method"},
    "customer": {"full_name", "email_address", "home_address"},  # Excludes bank_card for security
    # Add more objects and their attributes as needed
}

# Define restricted placeholder patterns or specific names if any
# For this example, we'll consider any placeholder not in VALID_PLACEHOLDERS as invalid.
# If there were specific patterns like `customer.password` to restrict, they could be checked explicitly.

def check_template_placeholders(template_string: str) -> bool:
    """
    Verifies that all placeholders in a template string are valid.
    Placeholders should be in the format {object.attribute}.

    Args:
        template_string: The template string to check.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any invalid or restricted placeholder is found.
    """
    # Regex to find placeholders like {object.attribute}
    # It captures the content inside the curly braces.
    placeholders_found = re.findall(r"\{([^}]+)\}", template_string)

    if not placeholders_found and "{" in template_string:
        # Handles cases like "{unclosed" or "{}"
        if re.search(r"\{[^{}]*\}", template_string) is None:
             raise ValueError("Malformed placeholder found in template.")


    for placeholder_content in placeholders_found:
        parts = placeholder_content.split('.')
        if len(parts) != 2:
            raise ValueError(
                f"Invalid placeholder format: '{placeholder_content}'. "
                "Expected format: '{object.attribute}'."
            )

        obj_type, attribute_name = parts[0], parts[1]

        if obj_type not in VALID_PLACEHOLDERS:
            raise ValueError(
                f"Invalid object type '{obj_type}' in placeholder '{{{placeholder_content}}}'. "
                f"Valid object types are: {', '.join(VALID_PLACEHOLDERS.keys())}."
            )

        if attribute_name not in VALID_PLACEHOLDERS[obj_type]:
            raise ValueError(
                f"Invalid attribute '{attribute_name}' for object type '{obj_type}' "
                f"in placeholder '{{{placeholder_content}}}'. "
                f"Valid attributes for '{obj_type}' are: {', '.join(VALID_PLACEHOLDERS[obj_type])}."
            )
        
        # Add any specific restricted checks here if needed
        # For example:
        # if obj_type == "customer" and attribute_name == "password":
        #     raise ValueError("Restricted placeholder: {customer.password} is not allowed.")

    return True


def compose_invoice_template(invoice_id: str, receipt_format: str) -> str:
    """
    Generates a formatted receipt by replacing placeholders in a template
    with actual order and customer details.

    Args:
        invoice_id: The ID of the invoice/order to use for details.
        receipt_format: The template string with placeholders.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the invoice_id is invalid, the template contains
                    invalid/restricted placeholders, or an attempt is made
                    to access protected customer information.
    """
    # 1. Validate template placeholders first.
    # This will raise ValueError if any placeholder is invalid or restricted
    # (e.g., {customer.bank_card} because "bank_card" is not in VALID_PLACEHOLDERS).
    check_template_placeholders(receipt_format)

    # 2. Retrieve order. get_order_by_id raises ValueError if not found.
    current_order = get_order_by_id(invoice_id)

    # 3. Define a replacer function for re.sub
    def replacer(match: re.Match[str]) -> str:
        placeholder_content = match.group(1)  # Content within {}
        parts = placeholder_content.split('.')
        # Format has been validated by check_template_placeholders, so parts[0] and parts[1] exist.
        obj_type, attribute_name = parts[0], parts[1]

        value: Any
        if obj_type == "order":
            # Attributes like 'client' are not directly substitutable as strings.
            # VALID_PLACEHOLDERS for 'order' should only contain simple types or lists.
            value = getattr(current_order, attribute_name)
            if attribute_name == "item_list":
                return ", ".join(map(str, value)) if isinstance(value, list) else str(value)
            elif attribute_name == "total_amount":
                return f"{float(value):.2f}"  # Format float to 2 decimal places
        elif obj_type == "customer":
            # 'bank_card' is not in VALID_PLACEHOLDERS['customer'], so check_template_placeholders
            # would have already raised an error if {customer.bank_card} was used.
            value = getattr(current_order.client, attribute_name)
        else:
            # This case should have been caught by check_template_placeholders
            # as an invalid object type.
            # Defensive coding, but ideally unreachable.
            raise ValueError(f"Internal error: Unknown object type '{obj_type}' after validation.")

        return str(value)

    # 4. Perform substitution
    formatted_receipt = re.sub(r"\{([^}]+)\}", replacer, receipt_format)
    return formatted_receipt


if __name__ == '__main__':
    print("--- Testing get_order_by_id ---")
    try:
        order1 = get_order_by_id("INV001")
        print(f"Found order INV001: Customer {order1.client.full_name}, Total: {order1.total_amount}")
        
        order_non_existent = get_order_by_id("INV999") # Non-existent
        print(f"Found order: {order_non_existent}")
    except ValueError as e:
        print(f"Error for get_order_by_id: {e}")

    print("\n--- Testing check_template_placeholders ---")
    templates_to_test = {
        "Valid Template 1": "Hello {customer.full_name}, your order {order.order_id} for ${order.total_amount} is ready. Items: {order.item_list}.",
        "Valid Template (No Placeholders)": "This is a static receipt.",
        "Invalid (Object Type)": "Details for {user.name}.", # 'user' is not a valid object type
        "Invalid (Attribute for order)": "Order amount: {order.client_name}.", # 'client_name' not in VALID_PLACEHOLDERS['order']
        "Invalid (Attribute for customer)": "Customer secret: {customer.bank_card}.", # 'bank_card' not in VALID_PLACEHOLDERS['customer']
        "Invalid (Malformed - Unclosed)": "Order: {order.order_id",
        "Invalid (Malformed - Empty)": "Value: {}",
        "Invalid (Format - Single Part)": "Data: {order}",
        "Invalid (Format - Too Many Parts)": "Data: {order.client.full_name}", # Only obj.attr supported by check_template_placeholders
    }

    for name, template in templates_to_test.items():
        print(f"\nTesting template: \"{template}\"")
        try:
            if check_template_placeholders(template):
                print(f"Result for '{name}': Valid")
        except ValueError as e:
            print(f"Result for '{name}': Invalid - {e}")

    print("\n--- Testing compose_invoice_template ---")
    # Test cases for compose_invoice_template
    valid_receipt_template = "Receipt for Order ID: {order.order_id}\nCustomer: {customer.full_name} ({customer.email_address})\nItems: {order.item_list}\nTotal Amount: ${order.total_amount}\nPayment Method: {order.payment_method}"
    
    test_compositions = [
        ("INV001", valid_receipt_template, "Valid composition"),
        ("INV002", "Dear {customer.full_name}, your total is ${order.total_amount}.", "Valid simple composition"),
        ("INV999", valid_receipt_template, "Composition with invalid Order ID"), # Should fail on get_order_by_id
        ("INV001", "Accessing protected: {customer.bank_card}", "Composition with restricted placeholder"), # Should fail on check_template_placeholders
        ("INV001", "Accessing invalid: {order.non_existent_attr}", "Composition with invalid placeholder attribute"), # Should fail on check_template_placeholders
    ]

    for inv_id, template_str, test_name in test_compositions:
        print(f"\nTesting composition: '{test_name}' (Order: {inv_id})")
        print(f"Template: \"{template_str}\"")
        try:
            receipt = compose_invoice_template(inv_id, template_str)
            print("Generated Receipt:")
            print(receipt)
        except ValueError as e:
            print(f"Error during composition: {e}")
