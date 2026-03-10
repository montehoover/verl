import re
from typing import Dict, Set, List # NamedTuple removed, List added for clarity

# Setup code provided by the user
class Buyer:
    def __init__(self, name: str, email: str, address: str, credit_card: str):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class PurchaseOrder:
    def __init__(self, id: str, customer: Buyer, items: List[str], total: float, payment_method: str):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

# Updated dictionary of purchase orders using the new classes
_purchase_orders: Dict[str, PurchaseOrder] = {
    "ORD123": PurchaseOrder("ORD123",
                            Buyer("Alice Black", "alice@example.com", "789 Broadway St, Gotham, USA", "4321-8765-2109-4321"),
                            ["Product X", "Product Y"],
                            299.50,
                            "Debit Card"),
    "ORD456": PurchaseOrder("ORD456",
                            Buyer("Bob White", "bob@example.com", "123 Main St, Metropolis, USA", "1111-2222-3333-4444"),
                            ["Gadget Z"],
                            19.99,
                            "Credit Card"),
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

# Define valid placeholders based on the new Buyer and PurchaseOrder classes.
VALID_PLACEHOLDERS: Set[str] = {
    "customer.name",        # Corresponds to PurchaseOrder.customer.name
    "customer.email",       # Corresponds to PurchaseOrder.customer.email
    "customer.address",     # Corresponds to PurchaseOrder.customer.address
    "customer.credit_card", # Corresponds to PurchaseOrder.customer.credit_card
    "order.id",             # Corresponds to PurchaseOrder.id
    "order.total",          # Corresponds to PurchaseOrder.total
    "order.items",          # Corresponds to PurchaseOrder.items
    "order.payment_method", # Corresponds to PurchaseOrder.payment_method
}

# Define restricted placeholders. Sensitive information should be restricted.
RESTRICTED_PLACEHOLDERS: Set[str] = {
    "customer.address",      # Example: Address might be sensitive for some receipts
    "customer.credit_card",  # Credit card information is highly sensitive
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

def generate_buyer_receipt(order_id: str, template: str) -> str:
    """
    Generates a formatted buyer receipt string using an order ID and a template.

    Args:
        order_id: The ID of the purchase order.
        template: The template string for the receipt.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid, or if the template uses
                    invalid or restricted placeholders.
    """
    # Validate placeholders first. This will raise ValueError if issues are found.
    check_template_placeholders(template)

    # Fetch the purchase order. This will raise ValueError if order_id is not found.
    order = get_purchase_order(order_id)

    # Prepare data for substitution
    # Note: Access to sensitive data like full address or credit_card is possible here
    # if not caught by check_template_placeholders. The check is crucial.
    placeholder_to_value = {
        "customer.name": order.customer.name,
        "customer.email": order.customer.email,
        # "customer.address" and "customer.credit_card" are handled by RESTRICTED_PLACEHOLDERS
        "order.id": order.id,
        "order.total": f"{order.total:.2f}", # Format total to 2 decimal places
        "order.items": ", ".join(order.items) if isinstance(order.items, list) else str(order.items),
        "order.payment_method": order.payment_method,
    }

    receipt = template
    # Find all placeholders in the template of the form {placeholder_key}
    found_placeholders_in_template = re.findall(r"\{([^}]+)\}", template)

    for ph_name in found_placeholders_in_template:
        # We only substitute placeholders that are explicitly mapped and are not restricted.
        # check_template_placeholders should have already validated them.
        if ph_name in placeholder_to_value:
            receipt = receipt.replace(f"{{{ph_name}}}", str(placeholder_to_value[ph_name]))
        # If ph_name is in VALID_PLACEHOLDERS but not in placeholder_to_value (e.g. restricted ones),
        # it won't be replaced here, relying on check_template_placeholders to have stopped it.
        # If it's an unknown placeholder, check_template_placeholders would also raise an error.

    return receipt

if __name__ == '__main__':
    # Example usage for get_purchase_order (updated for new data structure)
    print("--- Get Purchase Order Tests ---")
    try:
        order1 = get_purchase_order("ORD123")
        print(f"Found order: ID={order1.id}, Customer={order1.customer.name}, Total={order1.total}")

        order2 = get_purchase_order("ORD999") # This will raise ValueError
        print(f"Found order: {order2}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        order3 = get_purchase_order("ORD456")
        print(f"Found order: ID={order3.id}, Customer={order3.customer.name}, Total={order3.total}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Template Placeholder Checks (Updated Examples) ---")
    valid_template_po = "Receipt for {customer.name}. Order ID: {order.id}, Total: ${order.total}. Items: {order.items}."
    valid_template_email = "Dear {customer.name}, your order {order.id} for ${order.total} has been processed. Contact: {customer.email}."
    
    # This template now uses a restricted placeholder `customer.address`
    restricted_template_po = "Shipping to {customer.address} for {customer.name}."
    # This template uses a newly restricted placeholder `customer.credit_card`
    restricted_template_cc = "Payment with card {customer.credit_card} for order {order.id}."
    invalid_template_po = "Order {order.confirmation_code} for {customer.name}." # invalid placeholder

    templates_to_test = [
        ("Valid Template (Order Details)", valid_template_po),
        ("Valid Template (Email)", valid_template_email),
        ("Restricted Template (Address)", restricted_template_po),
        ("Restricted Template (Credit Card)", restricted_template_cc),
        ("Invalid Template (Unknown Placeholder)", invalid_template_po),
        ("Template without placeholders", "This is a static receipt."),
        ("Empty template", ""),
    ]

    for desc, template_str in templates_to_test:
        try:
            print(f"Checking template: \"{template_str}\" ({desc})")
            check_template_placeholders(template_str)
            print("Result: Valid")
        except ValueError as e:
            print(f"Result: Invalid - {e}")

    print("\n--- Generate Buyer Receipt Tests ---")
    receipt_template_success = "Thank you, {customer.name}! Your order {order.id} for ${order.total} including items: {order.items}, paid with {order.payment_method}, is confirmed. A confirmation will be sent to {customer.email}."
    receipt_template_fail_restricted = "Order {order.id} for {customer.name}. Deliver to {customer.address}." # Uses restricted placeholder
    receipt_template_fail_invalid_ph = "Order {order.id} for {customer.name}. Confirmation: {order.confirmation_no}." # Uses invalid placeholder

    # Test successful receipt generation
    try:
        print(f"\nGenerating receipt for ORD123 with template: \"{receipt_template_success}\"")
        receipt = generate_buyer_receipt("ORD123", receipt_template_success)
        print("Generated Receipt:\n" + receipt)
    except ValueError as e:
        print(f"Error generating receipt: {e}")

    # Test receipt generation with non-existent order ID
    try:
        print(f"\nGenerating receipt for ORD000 (non-existent) with template: \"{receipt_template_success}\"")
        receipt = generate_buyer_receipt("ORD000", receipt_template_success)
        print("Generated Receipt:\n" + receipt) # Should not reach here
    except ValueError as e:
        print(f"Error generating receipt: {e}")

    # Test receipt generation with restricted placeholder
    try:
        print(f"\nGenerating receipt for ORD123 with restricted template: \"{receipt_template_fail_restricted}\"")
        receipt = generate_buyer_receipt("ORD123", receipt_template_fail_restricted)
        print("Generated Receipt:\n" + receipt) # Should not reach here
    except ValueError as e:
        print(f"Error generating receipt: {e}")

    # Test receipt generation with invalid placeholder
    try:
        print(f"\nGenerating receipt for ORD123 with invalid placeholder template: \"{receipt_template_fail_invalid_ph}\"")
        receipt = generate_buyer_receipt("ORD123", receipt_template_fail_invalid_ph)
        print("Generated Receipt:\n" + receipt) # Should not reach here
    except ValueError as e:
        print(f"Error generating receipt: {e}")
