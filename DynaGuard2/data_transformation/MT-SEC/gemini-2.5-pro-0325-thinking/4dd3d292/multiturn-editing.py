import string

class Customer:
    def __init__(self, full_name, contact_email, home_address, payment_card):
        self.full_name = full_name
        self.contact_email = contact_email
        self.home_address = home_address  # Protected
        self.payment_card = payment_card  # Protected

class Order:
    def __init__(self, order_number, buyer: Customer, items_list, price_total, method_of_payment):
        self.order_number = order_number
        self.buyer = buyer
        self.items_list = items_list
        self.price_total = price_total
        self.method_of_payment = method_of_payment

purchases = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item A", "Item B"],
                    99.99,
                    "Credit Card"),
    "ORD002": Order("ORD002",
                    Customer("John Smith", "john@example.com", "456 Oak Ave, Otherville, USA", "9876-5432-1098-7654"),
                    ["Item C"],
                    49.50,
                    "PayPal")
}

# Define allowed and protected fields
PROTECTED_CUSTOMER_FIELDS = ["home_address", "payment_card"]

ALLOWED_DATA_ACCESS = {
    "customer.full_name": lambda o: o.buyer.full_name,
    "customer.contact_email": lambda o: o.buyer.contact_email,
    "order.order_number": lambda o: o.order_number,
    "order.items_list": lambda o: ", ".join(o.items_list),
    "order.price_total": lambda o: o.price_total,
    "order.method_of_payment": lambda o: o.method_of_payment,
}

def create_purchase_receipt(order_id: str, receipt_template: str) -> str:
    """
    Generates a purchase receipt string based on an order ID and a template.

    Args:
        order_id: The ID of the order.
        receipt_template: A string template with placeholders.
                          Examples: {customer.full_name}, {order.price_total}

    Returns:
        The generated receipt string.

    Raises:
        ValueError: If the order ID is invalid, or if the template attempts
                    to access protected customer information or uses invalid
                    placeholders.
    """
    order = purchases.get(order_id)
    if not order:
        raise ValueError(f"Invalid order ID: {order_id}")

    formatter = string.Formatter()
    placeholders_in_template = set()
    for _, field_name, _, _ in formatter.parse(receipt_template):
        if field_name is not None:
            placeholders_in_template.add(field_name)

    data_for_formatting = {}
    for placeholder in placeholders_in_template:
        # Check for attempts to access protected customer information
        if placeholder.startswith("customer."):
            attr_name = placeholder.split(".", 1)[1]
            if attr_name in PROTECTED_CUSTOMER_FIELDS:
                raise ValueError(f"Attempt to access protected customer information: {placeholder}")

        # Check if the placeholder is allowed and get its value
        if placeholder in ALLOWED_DATA_ACCESS:
            data_for_formatting[placeholder] = ALLOWED_DATA_ACCESS[placeholder](order)
        else:
            # This placeholder is not in protected list, and not in allowed list
            raise ValueError(f"Invalid placeholder: {placeholder}")
            
    try:
        return receipt_template.format(**data_for_formatting)
    except KeyError as e:
        # This should ideally be caught by the placeholder validation above,
        # but as a safeguard:
        raise ValueError(f"Error formatting receipt: Missing data for placeholder {e}")


if __name__ == "__main__":
    template1 = (
        "Receipt for Order: {order.order_number}\n"
        "Customer: {customer.full_name} ({customer.contact_email})\n"
        "Items: {order.items_list}\n"
        "Total: ${order.price_total:.2f}\n"
        "Paid via: {order.method_of_payment}"
    )

    print("--- Test Case 1: Valid Order ---")
    try:
        receipt1 = create_purchase_receipt("ORD001", template1)
        print(receipt1)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Case 2: Invalid Order ID ---")
    try:
        receipt2 = create_purchase_receipt("ORD999", template1)
        print(receipt2)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Case 3: Accessing Protected Customer Info (payment_card) ---")
    template_protected_card = "Customer Card: {customer.payment_card}"
    try:
        receipt3 = create_purchase_receipt("ORD001", template_protected_card)
        print(receipt3)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Case 4: Accessing Protected Customer Info (home_address) ---")
    template_protected_address = "Customer Address: {customer.home_address}"
    try:
        receipt4 = create_purchase_receipt("ORD001", template_protected_address)
        print(receipt4)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Case 5: Invalid Placeholder ---")
    template_invalid_placeholder = "Ship Date: {order.shipping_date}"
    try:
        receipt5 = create_purchase_receipt("ORD001", template_invalid_placeholder)
        print(receipt5)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Case 6: Valid Order ORD002 ---")
    try:
        receipt6 = create_purchase_receipt("ORD002", template1)
        print(receipt6)
    except ValueError as e:
        print(f"Error: {e}")
        
    print("\n--- Test Case 7: Template with no placeholders ---")
    template_no_placeholders = "This is a static receipt."
    try:
        receipt7 = create_purchase_receipt("ORD001", template_no_placeholders)
        print(receipt7)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Case 8: Template with only some allowed placeholders ---")
    template_partial = "Order: {order.order_number}, Total: ${order.price_total:.2f}"
    try:
        receipt8 = create_purchase_receipt("ORD001", template_partial)
        print(receipt8)
    except ValueError as e:
        print(f"Error: {e}")
