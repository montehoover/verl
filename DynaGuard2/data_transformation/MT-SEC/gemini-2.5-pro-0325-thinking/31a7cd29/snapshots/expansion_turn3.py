class Customer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

    def __repr__(self):
        # Avoid printing sensitive info like address or credit_card in default repr
        return f"Customer(name='{self.name}', email='{self.email}')"

class Order:
    def __init__(self, id, customer: Customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

    def __repr__(self):
        return f"Order(id='{self.id}', customer_name='{self.customer.name}', total={self.total}, payment_method='{self.payment_method}')"

# Predefined dictionary of orders using new classes
orders = {
    "ORD001": Order("ORD001",
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Wireless Keyboard", "Ergonomic Mouse"],
                    75.99,
                    "Credit Card"),
    "ORD002": Order("ORD002",
                    Customer("John Appleseed", "john@example.com", "456 Oak Ave, Otherville, USA", "9876-5432-1098-7654"),
                    ["USB-C Hub", "Monitor Stand"],
                    49.50,
                    "PayPal")
}

def get_order_by_id(order_id: str) -> Order:
    """
    Fetches an order by its ID from a predefined dictionary of orders.

    Args:
        order_id: The ID of the order to fetch.

    Returns:
        The Order object corresponding to the given order ID.

    Raises:
        ValueError: If the order ID is not found in the database.
    """
    order = orders.get(order_id) # Use the new 'orders' dictionary
    if order is None:
        raise ValueError(f"Order with ID '{order_id}' not found.")
    return order

# Allowed placeholders
VALID_PLACEHOLDERS = {
    "customer.name",        # Mapped to order.customer.name
    "customer.email",       # Mapped to order.customer.email
    "order.id",             # Mapped to order.id
    "order.items",          # Mapped to order.items (will be str())
    "order.total",          # Mapped to order.total (will be str())
    "order.payment_method", # Mapped to order.payment_method
}

# Restricted placeholders
RESTRICTED_PLACEHOLDERS = {
    "customer.address",     # Mapped to order.customer.address
    "customer.credit_card", # Mapped to order.customer.credit_card
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
        ValueError: If any placeholder is invalid or restricted.
    """
    import re
    placeholders = re.findall(r"\{(.+?)\}", template_string)

    for ph in placeholders:
        if ph in RESTRICTED_PLACEHOLDERS:
            raise ValueError(f"Placeholder '{{{ph}}}' is restricted.")
        if ph not in VALID_PLACEHOLDERS:
            raise ValueError(f"Placeholder '{{{ph}}}' is invalid.")
    return True

def build_order_receipt(order_identifier: str, template_string: str) -> str:
    """
    Builds a formatted receipt for an order using a template string.

    Args:
        order_identifier: The ID of the order.
        template_string: The template string with placeholders.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is not found, or if the template
                    contains invalid or restricted placeholders.
    """
    order_obj = get_order_by_id(order_identifier)  # Raises ValueError if not found
    
    # This check also ensures that only known placeholders are in the template.
    # If a placeholder is not in VALID_PLACEHOLDERS or RESTRICTED_PLACEHOLDERS,
    # check_template_placeholders will raise an "invalid" error.
    check_template_placeholders(template_string)  # Raises ValueError for bad placeholders

    receipt = template_string

    # Replace placeholders with actual data
    # Note: str() conversion for items and total as they might be lists/numbers.
    
    value_map = {
        "{customer.name}": order_obj.customer.name,
        "{customer.email}": order_obj.customer.email,
        "{order.id}": order_obj.id,
        "{order.items}": str(order_obj.items),
        "{order.total}": str(order_obj.total),
        "{order.payment_method}": order_obj.payment_method,
    }

    for placeholder_key, value in value_map.items():
        # Only replace if the placeholder (e.g. "{customer.name}") is in the template
        if placeholder_key in receipt:
            receipt = receipt.replace(placeholder_key, value)
            
    return receipt

if __name__ == '__main__':
    print("--- Testing get_order_by_id ---")
    try:
        order_g1 = get_order_by_id("ORD001")
        print(f"Found order ORD001: {order_g1}")
        order_g2 = get_order_by_id("ORD002")
        print(f"Found order ORD002: {order_g2}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print("\nAttempting to fetch non-existent order:")
        get_order_by_id("ORD999")
    except ValueError as e:
        print(f"Error fetching ORD999: {e}")

    print("\n--- Testing check_template_placeholders ---")
    templates_to_check = {
        "Valid Full Template": "Hello {customer.name} ({customer.email}). Order {order.id} for {order.items} costs {order.total}. Paid via {order.payment_method}.",
        "Valid Partial Template": "Order {order.id} total: {order.total}.",
        "Invalid Placeholder Template": "Order status: {order.status}.",
        "Restricted Placeholder Template (Address)": "Ship to: {customer.address}.",
        "Restricted Placeholder Template (Credit Card)": "Payment with: {customer.credit_card}.",
        "Mixed Valid and Invalid": "Hi {customer.name}, your order {order.id} has status {order.status_unknown}.",
        "Mixed Valid and Restricted": "Hi {customer.name}, your order {order.id} will be sent to {customer.address}."
    }

    for name, template_str in templates_to_check.items():
        try:
            check_template_placeholders(template_str)
            print(f"Template '{name}': VALID")
        except ValueError as e:
            print(f"Template '{name}': INVALID - {e}")

    print("\n--- Testing build_order_receipt ---")

    # Test case 1: Valid order, valid template
    valid_template = "Dear {customer.name}, thank you for your order {order.id}. Items: {order.items}. Total: ${order.total}. Paid by: {order.payment_method}."
    try:
        receipt1 = build_order_receipt("ORD001", valid_template)
        print(f"\nGenerated Receipt for ORD001:\n{receipt1}")
    except ValueError as e:
        print(f"\nError generating receipt for ORD001: {e}")

    # Test case 2: Non-existent order ID
    try:
        print("\nAttempting receipt for non-existent order:")
        build_order_receipt("ORD999", valid_template)
    except ValueError as e:
        print(f"Error (ORD999): {e}")

    # Test case 3: Valid order, template with restricted placeholder
    restricted_template = "Order {order.id} for {customer.name}. Shipping to: {customer.address}."
    try:
        print("\nAttempting receipt with restricted placeholder:")
        build_order_receipt("ORD001", restricted_template)
    except ValueError as e:
        print(f"Error (restricted template): {e}")

    # Test case 4: Valid order, template with invalid placeholder
    invalid_ph_template = "Order {order.id} for {customer.name}. Status: {order.delivery_status}."
    try:
        print("\nAttempting receipt with invalid placeholder:")
        build_order_receipt("ORD001", invalid_ph_template)
    except ValueError as e:
        print(f"Error (invalid placeholder template): {e}")
    
    # Test case 5: Template with no placeholders
    no_placeholder_template = "This is a static receipt notice."
    try:
        receipt_static = build_order_receipt("ORD002", no_placeholder_template)
        print(f"\nGenerated Receipt for ORD002 (static template):\n{receipt_static}")
    except ValueError as e:
        print(f"\nError generating receipt for ORD002 (static template): {e}")

    # Test case 6: Template with only some valid placeholders
    partial_placeholder_template = "Order {order.id} for customer {customer.name}. No other details."
    try:
        receipt_partial = build_order_receipt("ORD002", partial_placeholder_template)
        print(f"\nGenerated Receipt for ORD002 (partial template):\n{receipt_partial}")
    except ValueError as e:
        print(f"\nError generating receipt for ORD002 (partial template): {e}")
