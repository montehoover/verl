import re
from dataclasses import dataclass
from typing import Dict, Set

@dataclass
class Order:
    """Represents an order with an ID and details."""
    order_id: str
    customer_name: str
    total_amount: float
    items: list[str]

# Predefined dictionary of invoices (simulating a database or data source)
INVOICES: Dict[str, Order] = {
    "ORD123": Order(order_id="ORD123", customer_name="Alice Smith", total_amount=150.75, items=["Laptop", "Mouse"]),
    "ORD456": Order(order_id="ORD456", customer_name="Bob Johnson", total_amount=99.99, items=["Keyboard"]),
    "ORD789": Order(order_id="ORD789", customer_name="Carol Williams", total_amount=25.50, items=["USB Cable", "Charger"]),
}

def get_order_by_id(order_id: str) -> Order:
    """
    Retrieves an order by its ID from the predefined INVOICES.

    Args:
        order_id: The ID of the order to retrieve.

    Returns:
        The Order object if found.

    Raises:
        ValueError: If the order ID is not found in INVOICES.
    """
    order = INVOICES.get(order_id)
    if order is None:
        raise ValueError(f"Order with ID '{order_id}' not found.")
    return order

# Define valid placeholders
# This maps an object type (e.g., "customer", "order") to a set of its valid attributes.
VALID_PLACEHOLDERS: Dict[str, Set[str]] = {
    "order": {"order_id", "customer_name", "total_amount", "items"},
    "customer": {"full_name", "email", "address", "phone_number"},
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


if __name__ == '__main__':
    # Example usage for get_order_by_id:
    try:
        order1 = get_order_by_id("ORD123")
        print(f"Found order: {order1}")

        order2 = get_order_by_id("ORD456")
        print(f"Found order: {order2}")

        # Example of a non-existent order
        order3 = get_order_by_id("ORD000")
        print(f"Found order: {order3}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Testing get_order_by_id ---")
    try:
        # Another non-existent order to demonstrate the error
        order_non_existent = get_order_by_id("XYZ789")
        print(f"Found order: {order_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Testing check_template_placeholders ---")
    # Test cases for check_template_placeholders
    valid_template1 = "Hello {customer.full_name}, your order {order.order_id} for ${order.total_amount} is ready."
    valid_template2 = "Order for {customer.email}. Items: {order.items}."
    valid_template_no_placeholders = "This is a static receipt."

    invalid_template1 = "Invalid object: {user.name}" # Invalid object 'user'
    invalid_template2 = "Invalid attribute: {customer.age}" # Invalid attribute 'age' for 'customer'
    invalid_template3 = "Malformed: {order.total_amount" # Malformed placeholder (missing closing brace)
    invalid_template4 = "Invalid format: {order}" # Invalid format (not object.attribute)
    invalid_template5 = "Another invalid format: {order.total.amount}" # Too many parts
    invalid_template6 = "Empty placeholder: {}" # Empty placeholder
    invalid_template7 = "Text with {unclosed placeholder" 
    invalid_template8 = "Text with {customer.full_name} and {}."


    templates_to_test = {
        "Valid Template 1": valid_template1,
        "Valid Template 2": valid_template2,
        "Valid Template (No Placeholders)": valid_template_no_placeholders,
        "Invalid Template (Object)": invalid_template1,
        "Invalid Template (Attribute)": invalid_template2,
        "Invalid Template (Malformed - Unclosed)": invalid_template3, # This will be caught by regex not finding it, or by specific check
        "Invalid Template (Format - Single Part)": invalid_template4,
        "Invalid Template (Format - Too Many Parts)": invalid_template5,
        "Invalid Template (Empty Placeholder)": invalid_template6,
        "Invalid Template (Unclosed general)": invalid_template7,
        "Invalid Template (Valid and Empty)": invalid_template8,
    }

    for name, template in templates_to_test.items():
        print(f"\nTesting template: \"{template}\"")
        try:
            if check_template_placeholders(template):
                print(f"Result for '{name}': Valid")
        except ValueError as e:
            print(f"Result for '{name}': Invalid - {e}")
