import re
from typing import Dict, NamedTuple, Set

# Define a simple Order structure (can be a class or NamedTuple)
class Order(NamedTuple):
    order_id: str
    customer_name: str
    total_amount: float
    items: list[str]

# Predefined dictionary of orders
# In a real application, this would come from a database or an API
_ORDERS_DATABASE: Dict[str, Order] = {
    "order_123": Order(order_id="order_123", customer_name="Alice Smith", total_amount=150.75, items=["item_A", "item_B"]),
    "order_456": Order(order_id="order_456", customer_name="Bob Johnson", total_amount=99.99, items=["item_C"]),
    "order_789": Order(order_id="order_789", customer_name="Carol Williams", total_amount=230.00, items=["item_A", "item_D", "item_E"]),
}

def get_order_by_id(order_id: str) -> Order:
    """
    Fetches order details by order ID.

    Args:
        order_id: The ID of the order to fetch.

    Returns:
        The Order object if found.

    Raises:
        ValueError: If the order ID is not found in the database.
    """
    order = _ORDERS_DATABASE.get(order_id)
    if order is None:
        raise ValueError(f"Order with ID '{order_id}' not found.")
    return order

# Define a set of valid placeholders
VALID_PLACEHOLDERS: Set[str] = {
    "customer.full_name",
    "customer.first_name",
    "customer.last_name",
    "customer.email",
    "customer.address_line1",
    "customer.address_city",
    "customer.address_zip",
    "order.id",
    "order.date",
    "order.price_total",
    "order.items_count",
    "order.shipping_address_line1",
    "order.shipping_address_city",
    "order.shipping_address_zip",
    "shop.name",
    "shop.url",
}

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
    # Regex to find all placeholders like {placeholder.name}
    found_placeholders = re.findall(r'\{([^}]+)\}', template_string)

    for placeholder in found_placeholders:
        if placeholder not in VALID_PLACEHOLDERS:
            raise ValueError(f"Invalid or restricted placeholder found: {{{placeholder}}}")
    
    return True

if __name__ == '__main__':
    # Example usage for get_order_by_id:
    print("--- Testing get_order_by_id ---")
    try:
        order1 = get_order_by_id("order_123")
        print(f"Found order: {order1}")

        order2 = get_order_by_id("order_456")
        print(f"Found order: {order2}")

        # Example of a non-existent order
        order_non_existent = get_order_by_id("order_000")
        print(f"Found order: {order_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example demonstrating another non-existent order for get_order_by_id
    try:
        order_another_non_existent = get_order_by_id("order_xyz")
        print(f"Found order: {order_another_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Testing check_template_placeholders ---")
    # Example usage for check_template_placeholders:
    valid_template = "Hello {customer.full_name}, your order {order.id} for ${order.price_total} is confirmed."
    try:
        check_template_placeholders(valid_template)
        print(f"Template is valid: \"{valid_template}\"")
    except ValueError as e:
        print(f"Template validation error: {e}")

    invalid_template_1 = "Dear {customer.name}, your total is {order.total}." # customer.name and order.total are not in VALID_PLACEHOLDERS
    try:
        check_template_placeholders(invalid_template_1)
        print(f"Template is valid: \"{invalid_template_1}\"") # Should not reach here
    except ValueError as e:
        print(f"Template validation error: {e}")

    invalid_template_2 = "Order {order.id} contains {item.name}." # item.name is not valid
    try:
        check_template_placeholders(invalid_template_2)
        print(f"Template is valid: \"{invalid_template_2}\"") # Should not reach here
    except ValueError as e:
        print(f"Template validation error: {e}")
    
    empty_template = "This template has no placeholders."
    try:
        check_template_placeholders(empty_template)
        print(f"Template is valid: \"{empty_template}\"")
    except ValueError as e:
        print(f"Template validation error: {e}")

    template_with_only_valid_placeholders = "Your order {order.id} from {shop.name}."
    try:
        check_template_placeholders(template_with_only_valid_placeholders)
        print(f"Template is valid: \"{template_with_only_valid_placeholders}\"")
    except ValueError as e:
        print(f"Template validation error: {e}")
