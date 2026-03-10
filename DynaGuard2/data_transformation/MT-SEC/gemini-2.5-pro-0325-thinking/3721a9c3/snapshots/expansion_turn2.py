import re
from typing import Dict, NamedTuple, Set

class PurchaseOrder(NamedTuple):
    """Represents a purchase order."""
    order_id: str
    customer_name: str
    total_amount: float
    items: list

# Predefined dictionary of purchase orders
_purchase_orders: Dict[str, PurchaseOrder] = {
    "PO123": PurchaseOrder("PO123", "Alice Wonderland", 150.75, ["Book", "Tea Set"]),
    "PO456": PurchaseOrder("PO456", "Bob The Builder", 300.00, ["Tools", "Gloves"]),
    "PO789": PurchaseOrder("PO789", "Charlie Brown", 75.20, ["Kite", "Comic Book"]),
}

def get_purchase_order(order_id: str) -> PurchaseOrder:
    """
    Fetches purchase order details by order ID.

    Args:
        order_id: The ID of the purchase order to fetch.

    Returns:
        The PurchaseOrder object corresponding to the given order ID.

    Raises:
        ValueError: If the order ID is not found.
    """
    if order_id in _purchase_orders:
        return _purchase_orders[order_id]
    else:
        raise ValueError(f"Purchase order with ID '{order_id}' not found.")

# Define valid placeholders. These correspond to attributes of PurchaseOrder or related concepts.
# For simplicity, let's assume placeholders are in the format {object.attribute}
VALID_PLACEHOLDERS: Set[str] = {
    "customer.name",  # Corresponds to PurchaseOrder.customer_name
    "order.id",       # Corresponds to PurchaseOrder.order_id
    "order.total",    # Corresponds to PurchaseOrder.total_amount
    "order.items",    # Corresponds to PurchaseOrder.items
}

# Define restricted placeholders (if any). For this example, let's imagine 'customer.address' is restricted.
RESTRICTED_PLACEHOLDERS: Set[str] = {
    "customer.address",
}

def check_template_placeholders(template_string: str) -> bool:
    """
    Verifies that all placeholders in a template string are valid and not restricted.

    Placeholders are expected in the format {placeholder_name}, e.g., {customer.name}.

    Args:
        template_string: The template string containing placeholders.

    Returns:
        True if all placeholders are valid and not restricted.

    Raises:
        ValueError: If any placeholder is invalid or restricted.
    """
    # Regex to find placeholders like {placeholder.name}
    found_placeholders = re.findall(r"\{([^}]+)\}", template_string)

    if not found_placeholders and template_string: # Template has content but no placeholders
        return True
    if not found_placeholders and not template_string: # Empty template
        return True


    for placeholder in found_placeholders:
        if placeholder in RESTRICTED_PLACEHOLDERS:
            raise ValueError(f"Placeholder '{{{placeholder}}}' is restricted.")
        if placeholder not in VALID_PLACEHOLDERS:
            raise ValueError(f"Placeholder '{{{placeholder}}}' is invalid.")
    
    return True

if __name__ == '__main__':
    # Example usage for get_purchase_order:
    try:
        order1 = get_purchase_order("PO123")
        print(f"Found order: {order1}")

        order2 = get_purchase_order("PO999") # This will raise ValueError
        print(f"Found order: {order2}")
    except ValueError as e:
        print(e)

    try:
        order3 = get_purchase_order("PO456")
        print(f"Found order: {order3}")
    except ValueError as e:
        print(e)

    print("\n--- Template Placeholder Checks ---")
    # Example usage for check_template_placeholders:
    valid_template1 = "Receipt for {customer.name}. Order ID: {order.id}, Total: {order.total}."
    valid_template2 = "Items: {order.items} for order {order.id}."
    invalid_template1 = "Dear {customer.name}, your order {order.number} is ready." # invalid placeholder
    restricted_template1 = "Send to {customer.address} for {customer.name}." # restricted placeholder
    template_without_placeholders = "This is a static receipt."
    empty_template = ""

    try:
        print(f"Checking template: \"{valid_template1}\"")
        check_template_placeholders(valid_template1)
        print("Result: Valid")
    except ValueError as e:
        print(f"Result: Invalid - {e}")

    try:
        print(f"Checking template: \"{valid_template2}\"")
        check_template_placeholders(valid_template2)
        print("Result: Valid")
    except ValueError as e:
        print(f"Result: Invalid - {e}")

    try:
        print(f"Checking template: \"{invalid_template1}\"")
        check_template_placeholders(invalid_template1)
        print("Result: Valid") # Should not reach here
    except ValueError as e:
        print(f"Result: Invalid - {e}")

    try:
        print(f"Checking template: \"{restricted_template1}\"")
        check_template_placeholders(restricted_template1)
        print("Result: Valid") # Should not reach here
    except ValueError as e:
        print(f"Result: Invalid - {e}")
    
    try:
        print(f"Checking template: \"{template_without_placeholders}\"")
        check_template_placeholders(template_without_placeholders)
        print("Result: Valid")
    except ValueError as e:
        print(f"Result: Invalid - {e}")

    try:
        print(f"Checking template: \"{empty_template}\"")
        check_template_placeholders(empty_template)
        print("Result: Valid")
    except ValueError as e:
        print(f"Result: Invalid - {e}")
