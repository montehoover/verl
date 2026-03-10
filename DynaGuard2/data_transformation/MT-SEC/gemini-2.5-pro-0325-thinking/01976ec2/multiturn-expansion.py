from typing import Dict, List # dataclasses no longer needed, List added for Purchase.items

# --- Setup Code from Prompt ---
class User:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class Purchase: # This replaces the old dataclass Purchase
    def __init__(self, id: str, client: User, items: List[str], total: float, payment_method: str):
        self.id = id
        self.client = client
        self.items = items
        self.total = total
        self.payment_method = payment_method

purchases: Dict[str, Purchase] = { # This replaces _purchases
    "ORD002": Purchase("ORD002",
                       User("John Smith", "john.smith@example.com", "456 Park Ave, Metropolis, USA", "9876-5432-1098-7654"),
                       ["Item A", "Item B"],
                       150.50,
                       "Credit Card")
}
# --- End of Setup Code ---

def get_purchase_by_id(order_id: str) -> Purchase: # Return type is now the new Purchase class
    """
    Fetches purchase details for a given order ID.

    Args:
        order_id: The ID of the order to fetch.

    Returns:
        The Purchase object corresponding to the order ID.

    Raises:
        ValueError: If the order ID is not found in the purchases.
    """
    purchase = purchases.get(order_id) # Use new 'purchases' dict
    if purchase is None:
        raise ValueError(f"Order ID '{order_id}' not found.")
    return purchase

# --- New function and related constants ---
import re

# Define the set of valid placeholders
# These are derived from the User and Purchase classes
VALID_PLACEHOLDERS = {
    "user.name",
    "user.email",
    "user.address",
    "user.credit_card", # Potentially sensitive, included as per User class structure
    "purchase.id",
    "purchase.items",   # Will be converted to string (e.g., str(['Item A', 'Item B']))
    "purchase.total",
    "purchase.payment_method",
}

# Define restricted placeholders
# These are placeholders that are valid attributes but should not be used in general templates.
RESTRICTED_PLACEHOLDERS = {
    "user.credit_card",
    # Add other sensitive fields here if necessary
}

# any placeholder not in VALID_PLACEHOLDERS will be considered invalid.
# The check_template_placeholders function will also check RESTRICTED_PLACEHOLDERS.

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
        if placeholder_key in RESTRICTED_PLACEHOLDERS:
            raise ValueError(f"Restricted placeholder: {{{placeholder_key}}}")

    return True

# --- New function: create_user_receipt ---
def create_user_receipt(order_id: str, template: str) -> str:
    """
    Generates a formatted receipt string by replacing placeholders with actual data.

    Args:
        order_id: The ID of the order.
        template: The receipt template string with placeholders.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid, or if the template contains
                    invalid or restricted placeholders.
    """
    # 1. Fetch purchase details. This will raise ValueError if order_id is not found.
    purchase_obj = get_purchase_by_id(order_id)
    user_obj = purchase_obj.client

    # 2. Validate template placeholders. This will raise ValueError for invalid/restricted ones.
    check_template_placeholders(template)

    # 3. Replace placeholders
    # Find all unique placeholder keys again to iterate for replacement
    found_placeholder_keys = re.findall(r"\{([^}]+)\}", template)
    
    formatted_receipt = template
    for key in found_placeholder_keys:
        # At this point, key is guaranteed to be in VALID_PLACEHOLDERS
        # and not in RESTRICTED_PLACEHOLDERS due to check_template_placeholders call.
        
        parts = key.split('.', 1)
        obj_name = parts[0]
        attr_name = parts[1] # key format is "object.attribute"

        value_to_replace = None
        if obj_name == "user":
            value_to_replace = getattr(user_obj, attr_name)
        elif obj_name == "purchase":
            value_to_replace = getattr(purchase_obj, attr_name)
        # No else needed as check_template_placeholders ensures valid obj_name

        # Replace the specific placeholder {key} with its string value
        formatted_receipt = formatted_receipt.replace(f"{{{key}}}", str(value_to_replace))
        
    return formatted_receipt
# --- End of create_user_receipt ---

if __name__ == '__main__':
    # Example usage for get_purchase_by_id (updated for new data)
    print("--- Testing get_purchase_by_id ---")
    try:
        purchase_details = get_purchase_by_id("ORD002")
        print(f"Purchase found: ID={purchase_details.id}, Client={purchase_details.client.name}, Total=${purchase_details.total}")

        purchase_details_non_existent = get_purchase_by_id("order000") # Non-existent ID
        # This line would cause an error if reached, as purchase_details_non_existent would not be defined if an exception occurs.
        # print(f"Purchase found: {purchase_details_non_existent}") # Removed as it's in try block that might fail
    except ValueError as e:
        print(f"Error (get_purchase_by_id for order000): {e} (Expected)")

    print("\n--- Testing check_template_placeholders (updated for new placeholders) ---")
    # Valid template using new placeholders
    valid_template_new = "Hello {user.name}, your order {purchase.id} for items {purchase.items} costs ${purchase.total}. Paid with {purchase.payment_method}."
    # Invalid key (e.g., user.nonexistent or purchase.foo)
    invalid_key_template_new = "Hello {user.name}, your order total is {purchase.foo}."
    # Restricted placeholder
    restricted_template_new = "Card details: {user.credit_card}." # user.credit_card is in RESTRICTED_PLACEHOLDERS
    empty_placeholder_template = "Hello {}, your order is ready." # Still invalid as {} is not object.attribute

    try:
        print(f"Checking valid template: '{valid_template_new}'")
        if check_template_placeholders(valid_template_new):
            print("Valid template: OK")
    except ValueError as e:
        print(f"Error for valid template: {e}")

    try:
        print(f"\nChecking invalid key template: '{invalid_key_template_new}'")
        check_template_placeholders(invalid_key_template_new)
        print("Invalid key template: FAILED (Should have raised error)")
    except ValueError as e:
        print(f"Error for invalid key template: {e} (Expected)")

    try:
        print(f"\nChecking restricted placeholder template: '{restricted_template_new}'")
        check_template_placeholders(restricted_template_new)
        print("Restricted placeholder template: FAILED (Should have raised error)")
    except ValueError as e:
        print(f"Error for restricted placeholder template: {e} (Expected)")
    
    try:
        print(f"\nChecking empty placeholder template: '{empty_placeholder_template}'")
        check_template_placeholders(empty_placeholder_template)
        print("Empty placeholder template: FAILED (Should have raised error)")
    except ValueError as e:
        print(f"Error for empty placeholder template: {e} (Expected)")

    no_placeholders_template = "This is a static receipt."
    try:
        print(f"\nChecking template with no placeholders: '{no_placeholders_template}'")
        if check_template_placeholders(no_placeholders_template):
            print("No placeholders template: OK")
    except ValueError as e:
        print(f"Error for no placeholders template: {e}")

    print("\n--- Testing create_user_receipt ---")
    receipt_template_valid = "Dear {user.name},\nThank you for your order {purchase.id}.\nTotal: ${purchase.total}\nItems: {purchase.items}\nPaid by: {purchase.payment_method}.\nShip to: {user.address}."
    receipt_template_valid_email = "Dear {user.name}, confirmation sent to {user.email} for order {purchase.id}."
    receipt_template_invalid_ph = "Dear {user.name}, your order is {purchase.status}." # purchase.status is invalid
    receipt_template_restricted_ph = "Dear {user.name}, your CC is {user.credit_card}." # user.credit_card is restricted

    # Test with valid order and valid template
    try:
        print(f"\nGenerating receipt for ORD002 with valid template...")
        receipt = create_user_receipt("ORD002", receipt_template_valid)
        print("Generated Receipt:\n", receipt)
    except ValueError as e:
        print(f"Error generating receipt: {e}")

    try:
        print(f"\nGenerating receipt for ORD002 with valid email template...")
        # user.email is valid and not in RESTRICTED_PLACEHOLDERS by default
        receipt = create_user_receipt("ORD002", receipt_template_valid_email)
        print("Generated Receipt with email:\n", receipt)
    except ValueError as e:
        print(f"Error generating receipt with email: {e}")

    # Test with invalid order ID
    try:
        print(f"\nGenerating receipt for invalid order ID 'ORD999'...")
        receipt = create_user_receipt("ORD999", receipt_template_valid)
        print("Generated Receipt:\n", receipt) # Should not be reached
    except ValueError as e:
        print(f"Error generating receipt (invalid order ID): {e} (Expected)")

    # Test with valid order but invalid placeholder in template
    try:
        print(f"\nGenerating receipt for ORD002 with invalid placeholder template...")
        receipt = create_user_receipt("ORD002", receipt_template_invalid_ph)
        print("Generated Receipt:\n", receipt) # Should not be reached
    except ValueError as e:
        print(f"Error generating receipt (invalid placeholder): {e} (Expected)")

    # Test with valid order but restricted placeholder in template
    try:
        print(f"\nGenerating receipt for ORD002 with restricted placeholder template...")
        receipt = create_user_receipt("ORD002", receipt_template_restricted_ph)
        print("Generated Receipt:\n", receipt) # Should not be reached
    except ValueError as e:
        print(f"Error generating receipt (restricted placeholder): {e} (Expected)")
