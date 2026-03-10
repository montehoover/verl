class Order:
    """Represents an order with an ID and details."""
    def __init__(self, order_id, details):
        self.order_id = order_id
        self.details = details

    def __repr__(self):
        return f"Order(order_id='{self.order_id}', details='{self.details}')"

# Predefined dictionary of orders
_orders_database = {
    "101": Order("101", "Order for 2x Widget A"),
    "102": Order("102", "Order for 1x Gadget B"),
    "103": Order("103", "Order for 5x Gizmo C"),
}

def get_order_by_id(order_id: str) -> Order:
    """
    Retrieves an order by its ID from a predefined dictionary.

    Args:
        order_id: The ID of the order to retrieve.

    Returns:
        The Order object corresponding to the given ID.

    Raises:
        ValueError: If the order ID does not exist in the database.
    """
    order = _orders_database.get(order_id)
    if order is None:
        raise ValueError(f"Order with ID '{order_id}' not found.")
    return order

# --- Template Placeholder Validation ---

import re

# Define allowed placeholders and sensitive attributes
# These would typically be more extensive and configurable in a real application
VALID_OBJECT_ATTRIBUTES = {
    "customer": {"name", "address", "email"},
    "order": {"id", "total", "date", "items"},
    "item": {"name", "quantity", "price"},
}

SENSITIVE_ATTRIBUTES = {
    "customer.password_hash", # Example of a sensitive field
    "order.payment_details_token", # Another example
}

def check_template_placeholders(template_string: str) -> bool:
    """
    Verifies that all placeholders in a template string are valid and not sensitive.

    Placeholders should be in the format {object.attribute}.

    Args:
        template_string: The template string to check.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid, improperly formatted,
                    or attempts to access sensitive information.
    """
    placeholders = re.findall(r"\{(.+?)\}", template_string)

    for ph_full in placeholders:
        if '.' not in ph_full:
            raise ValueError(
                f"Invalid placeholder format: '{ph_full}'. "
                "Expected format 'object.attribute'."
            )

        obj_name, attr_name = ph_full.split('.', 1)

        if obj_name not in VALID_OBJECT_ATTRIBUTES:
            raise ValueError(
                f"Invalid object '{obj_name}' in placeholder '{{{ph_full}}}'. "
                f"Allowed objects are: {', '.join(VALID_OBJECT_ATTRIBUTES.keys())}."
            )

        if attr_name not in VALID_OBJECT_ATTRIBUTES[obj_name]:
            raise ValueError(
                f"Invalid attribute '{attr_name}' for object '{obj_name}' "
                f"in placeholder '{{{ph_full}}}'. Allowed attributes for '{obj_name}' "
                f"are: {', '.join(VALID_OBJECT_ATTRIBUTES[obj_name])}."
            )

        if ph_full in SENSITIVE_ATTRIBUTES:
            raise ValueError(
                f"Attempt to access sensitive attribute "
                f"'{ph_full}' in template."
            )
        # Could also check obj_name.attr_name against a pattern for sensitive attributes
        # e.g., if attr_name.endswith("_token") or attr_name == "password"

    return True


if __name__ == '__main__':
    # Example usage for get_order_by_id:
    print("--- Testing get_order_by_id ---")
    try:
        order1 = get_order_by_id("101")
        print(f"Found order: {order1}")

        order2 = get_order_by_id("102")
        print(f"Found order: {order2}")

        # Example of a non-existent order
        order_non_existent = get_order_by_id("999")
        print(f"Found order: {order_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent order to show the error again
        order_fake = get_order_by_id("abc")
        print(f"Found order: {order_fake}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Testing check_template_placeholders ---")
    # Example usage for check_template_placeholders:
    valid_template_1 = "Hello {customer.name}, your order {order.id} for ${order.total} is confirmed."
    valid_template_2 = "Items: {item.name} x {item.quantity}."
    invalid_format_template = "Dear {customer_name}, your order total is {order.total}." # Missing dot
    invalid_object_template = "User: {user.name}, Order: {order.id}." # Invalid object 'user'
    invalid_attribute_template = "Order {order.id}, Status: {order.status}." # Invalid attribute 'status' for 'order'
    sensitive_template = "Customer: {customer.name}, Secret: {customer.password_hash}."

    templates_to_test = {
        "Valid Template 1": valid_template_1,
        "Valid Template 2": valid_template_2,
        "Invalid Format Template": invalid_format_template,
        "Invalid Object Template": invalid_object_template,
        "Invalid Attribute Template": invalid_attribute_template,
        "Sensitive Data Template": sensitive_template,
        "Mixed Valid/Invalid Template": "Hi {customer.name}, your {order.payment_details_token} is processed.",
        "Valid with unknown object": "Details: {product.name}" # product not in VALID_OBJECT_ATTRIBUTES
    }

    for name, template in templates_to_test.items():
        print(f"\nTesting template: '{name}'")
        print(f"Template string: \"{template}\"")
        try:
            if check_template_placeholders(template):
                print("Result: Template is VALID.")
        except ValueError as e:
            print(f"Result: Template is INVALID. Reason: {e}")
