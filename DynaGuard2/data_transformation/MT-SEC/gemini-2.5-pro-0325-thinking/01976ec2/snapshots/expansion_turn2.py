from dataclasses import dataclass
from typing import Dict

@dataclass
class Purchase:
    """Represents a purchase made by a customer."""
    item_id: str
    price: float
    customer_id: str

# Predefined dictionary of purchases, with order_id as the key
_purchases: Dict[str, Purchase] = {
    "order123": Purchase(item_id="itemA", price=19.99, customer_id="cust001"),
    "order456": Purchase(item_id="itemB", price=29.50, customer_id="cust002"),
    "order789": Purchase(item_id="itemC", price=9.75, customer_id="cust001"),
}

def get_purchase_by_id(order_id: str) -> Purchase:
    """
    Fetches purchase details for a given order ID.

    Args:
        order_id: The ID of the order to fetch.

    Returns:
        The Purchase object corresponding to the order ID.

    Raises:
        ValueError: If the order ID is not found in the purchases.
    """
    purchase = _purchases.get(order_id)
    if purchase is None:
        raise ValueError(f"Order ID '{order_id}' not found.")
    return purchase

# --- New function and related constants ---
import re

# Define the set of valid placeholders
# These would typically be derived from your data models (e.g., Customer, Order, Item)
VALID_PLACEHOLDERS = {
    "customer.name",
    "customer.id",  # Corresponds to Purchase.customer_id
    "order.id",     # Corresponds to the order_id key
    "order.total",  # Assumed to be derivable, e.g., Purchase.price for single-item orders
    "item.name",    # Corresponds to Purchase.item_id (or a looked-up name)
    "item.price",   # Corresponds to Purchase.price
}

# No explicitly restricted placeholders defined for now,
# any placeholder not in VALID_PLACEHOLDERS will be considered invalid.

def check_template_placeholders(template_string: str) -> bool:
    """
    Verifies that all placeholders in a template string are valid.

    Placeholders are expected in the format {object.attribute}.

    Args:
        template_string: The template string to check.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any invalid or restricted placeholder is found.
    """
    # Find all placeholders in the format {placeholder_key}
    found_placeholders = re.findall(r"\{([^}]+)\}", template_string)

    if not found_placeholders and "{" in template_string and "}" in template_string:
        # Handles cases like "{}" or "{invalid format" which re.findall might miss
        # depending on the exact regex, but r"\{([^}]+)\}" is robust for valid formats.
        # This check is more for malformed placeholders.
        # A simple check for any placeholder-like syntax that wasn't caught.
        if re.search(r"\{[^\w.]*\}", template_string) or not re.search(r"\{[\w.]+\}", template_string):
             # Check for empty {} or malformed ones not matching `object.attribute`
            if "{}" in template_string:
                 raise ValueError("Empty placeholder '{}' found.")
            # This part might need more sophisticated parsing for truly malformed placeholders
            # For now, we assume placeholders are either well-formed {key} or not present.

    for placeholder_key in found_placeholders:
        if placeholder_key not in VALID_PLACEHOLDERS:
            raise ValueError(f"Invalid placeholder: {{{placeholder_key}}}")
        # Add checks for restricted placeholders here if needed
        # For example:
        # if placeholder_key in RESTRICTED_PLACEHOLDERS:
        #     raise ValueError(f"Restricted placeholder: {{{placeholder_key}}}")

    return True

if __name__ == '__main__':
    # Example usage:
    try:
        purchase_details = get_purchase_by_id("order123")
        print(f"Purchase found: {purchase_details}")

        purchase_details_non_existent = get_purchase_by_id("order000")
        print(f"Purchase found: {purchase_details_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        purchase_details_456 = get_purchase_by_id("order456")
        print(f"Purchase found: {purchase_details_456}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Testing check_template_placeholders ---")
    valid_template = "Hello {customer.name}, your order {order.id} for {item.name} costs ${item.price} (Total: ${order.total}). Customer ID: {customer.id}."
    invalid_template_key = "Hello {customer.name}, your order total is {order.sum}." # order.sum is invalid
    invalid_template_format = "Hello {customer.name}, your order has an {invalid format}."
    empty_placeholder_template = "Hello {}, your order is ready."

    try:
        print(f"Checking valid template: '{valid_template}'")
        if check_template_placeholders(valid_template):
            print("Valid template: OK")
    except ValueError as e:
        print(f"Error for valid template: {e}")

    try:
        print(f"\nChecking invalid key template: '{invalid_template_key}'")
        check_template_placeholders(invalid_template_key)
        print("Invalid key template: OK (Should have raised error)")
    except ValueError as e:
        print(f"Error for invalid key template: {e} (Expected)")

    try:
        print(f"\nChecking invalid format template: '{invalid_template_format}'")
        # Note: "{invalid format}" will be extracted as "invalid format" by the regex.
        # If "invalid format" is not in VALID_PLACEHOLDERS, it will be caught.
        check_template_placeholders(invalid_template_format)
        print("Invalid format template: OK (Should have raised error)")
    except ValueError as e:
        print(f"Error for invalid format template: {e} (Expected)")
    
    try:
        print(f"\nChecking empty placeholder template: '{empty_placeholder_template}'")
        check_template_placeholders(empty_placeholder_template)
        print("Empty placeholder template: OK (Should have raised error)")
    except ValueError as e:
        print(f"Error for empty placeholder template: {e} (Expected)")

    no_placeholders_template = "This is a static receipt."
    try:
        print(f"\nChecking template with no placeholders: '{no_placeholders_template}'")
        if check_template_placeholders(no_placeholders_template):
            print("No placeholders template: OK")
    except ValueError as e:
        print(f"Error for no placeholders template: {e}")
