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

# --- New classes and data for create_purchase_receipt ---
class ReceiptCustomer:
    def __init__(self, full_name: str, contact_email: str, home_address: str, payment_card: str):
        self.full_name = full_name
        self.contact_email = contact_email
        self.home_address = home_address # Expected format: "Street, City, ZIP" or "Street, City, State/Country"
        self.payment_card = payment_card # Note: This is sensitive data, handle with care in a real app.

class ReceiptOrder:
    def __init__(self, order_number: str, buyer: ReceiptCustomer, items_list: list[str], price_total: float, method_of_payment: str):
        self.order_number = order_number
        self.buyer = buyer
        self.items_list = items_list
        self.price_total = price_total
        self.method_of_payment = method_of_payment

receipt_purchases: Dict[str, ReceiptOrder] = {
    "ORD001": ReceiptOrder(
        order_number="ORD001",
        buyer=ReceiptCustomer(
            full_name="Jane Doe",
            contact_email="jane@example.com",
            home_address="123 Main St, Anytown, USA", # Example format
            payment_card="1234-5678-9012-3456" # Example, sensitive
        ),
        items_list=["Item A", "Item B"],
        price_total=99.99,
        method_of_payment="Credit Card"
    ),
    "ORD002": ReceiptOrder(
        order_number="ORD002",
        buyer=ReceiptCustomer(
            full_name="John Smith",
            contact_email="john@example.com",
            home_address="456 Oak Ave, Otherville, CA 90210", # Example format with ZIP
            payment_card="9876-5432-1098-7654"
        ),
        items_list=["Item C", "Item D", "Item E"],
        price_total=245.50,
        method_of_payment="PayPal"
    )
}

def create_purchase_receipt(order_id: str, receipt_template: str) -> str:
    """
    Generates a formatted purchase receipt string by replacing placeholders
    in a template with actual order and customer details.

    Args:
        order_id: The ID of the order to generate the receipt for.
        receipt_template: The template string containing placeholders.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is not found, if the template contains
                    invalid/restricted placeholders, or if a valid placeholder
                    cannot be resolved with available data.
    """
    # 1. Validate template placeholders against the global VALID_PLACEHOLDERS set
    check_template_placeholders(receipt_template)

    # 2. Fetch the order object
    order_obj = receipt_purchases.get(order_id)
    if order_obj is None:
        raise ValueError(f"Order with ID '{order_id}' not found in receipt_purchases.")

    customer_obj = order_obj.buyer

    # 3. Prepare data for template rendering
    data_for_template: Dict[str, str] = {}
    
    # Customer details
    data_for_template["customer.full_name"] = customer_obj.full_name
    name_parts = customer_obj.full_name.split(" ", 1)
    data_for_template["customer.first_name"] = name_parts[0]
    data_for_template["customer.last_name"] = name_parts[1] if len(name_parts) > 1 else ""
    data_for_template["customer.email"] = customer_obj.contact_email
    
    addr_parts = [p.strip() for p in customer_obj.home_address.split(',')]
    data_for_template["customer.address_line1"] = addr_parts[0] if len(addr_parts) > 0 else ""
    data_for_template["customer.address_city"] = addr_parts[1] if len(addr_parts) > 1 else ""
    # Assuming the last part of home_address is zip/state/country based on VALID_PLACEHOLDERS
    data_for_template["customer.address_zip"] = addr_parts[2] if len(addr_parts) > 2 else (addr_parts[-1] if len(addr_parts) > 0 else "")


    # Order details
    data_for_template["order.id"] = order_obj.order_number
    data_for_template["order.price_total"] = f"{order_obj.price_total:.2f}" # Format price to 2 decimal places
    data_for_template["order.items_count"] = str(len(order_obj.items_list))

    # Note: Placeholders like 'order.date', 'shop.name', 'order.shipping_address_line1'
    # are in VALID_PLACEHOLDERS but not directly available in ReceiptOrder/ReceiptCustomer.
    # The replacer function below will raise an error if such placeholders are used.

    # 4. Define replacer function for re.sub
    def replacer(match: re.Match) -> str:
        placeholder = match.group(1)
        if placeholder in data_for_template:
            return data_for_template[placeholder]
        # This case handles placeholders that are in VALID_PLACEHOLDERS (so check_template_placeholders passed them)
        # but for which we don't have data from the ReceiptOrder/ReceiptCustomer models.
        elif placeholder in VALID_PLACEHOLDERS:
            raise ValueError(
                f"Placeholder {{{placeholder}}} is valid but its data is not available "
                f"for order {order_id} (e.g., 'order.date', 'shop.name', 'order.shipping_address_line1' are not populated by this function)."
            )
        # This else block should ideally not be reached if check_template_placeholders works correctly,
        # as it would mean a placeholder not in VALID_PLACEHOLDERS was found.
        # For robustness, we can return the original match or raise an error.
        # However, check_template_placeholders should have already raised an error.
        # If it's reached, it implies an issue with check_template_placeholders or logic.
        # To be safe, let's ensure it's an error if somehow a non-validated placeholder gets here.
        raise ValueError(f"Unexpected placeholder {{{placeholder}}} encountered during replacement. It should have been caught by check_template_placeholders.")

    # 5. Substitute placeholders in the template
    filled_receipt = re.sub(r'\{([^}]+)\}', replacer, receipt_template)
    
    return filled_receipt

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

    template_with_only_valid_placeholders = "Your order {order.id} from {shop.name}." # shop.name will cause issues later if used with create_purchase_receipt
    try:
        check_template_placeholders(template_with_only_valid_placeholders)
        print(f"Template is valid (syntax check): \"{template_with_only_valid_placeholders}\"")
    except ValueError as e:
        print(f"Template validation error: {e}")

    print("\n--- Testing create_purchase_receipt ---")
    receipt_template_valid = (
        "Receipt for Order: {order.id}\n"
        "Customer: {customer.full_name} ({customer.email})\n"
        "Address: {customer.address_line1}, {customer.address_city}, {customer.address_zip}\n"
        "Total Amount: ${order.price_total}\n"
        "Number of Items: {order.items_count}\n"
        "Thank you for your purchase, {customer.first_name}!"
    )
    try:
        receipt = create_purchase_receipt("ORD001", receipt_template_valid)
        print("\nGenerated Receipt (ORD001):")
        print(receipt)
    except ValueError as e:
        print(f"Error generating receipt for ORD001: {e}")

    try:
        receipt_ord002 = create_purchase_receipt("ORD002", receipt_template_valid)
        print("\nGenerated Receipt (ORD002):")
        print(receipt_ord002)
    except ValueError as e:
        print(f"Error generating receipt for ORD002: {e}")

    # Test with an invalid order ID
    try:
        create_purchase_receipt("ORD999", receipt_template_valid)
    except ValueError as e:
        print(f"\nError (expected for invalid order ID): {e}")

    # Test with a template having a restricted placeholder (should be caught by check_template_placeholders)
    receipt_template_invalid_placeholder = "Your order is {order.secret_details}."
    try:
        create_purchase_receipt("ORD001", receipt_template_invalid_placeholder)
    except ValueError as e:
        print(f"\nError (expected for restricted placeholder): {e}")

    # Test with a template having a valid placeholder that cannot be resolved by create_purchase_receipt's data model
    receipt_template_unresolvable_placeholder = "Order Date: {order.date}, Shop: {shop.name}, Order ID: {order.id}"
    try:
        # check_template_placeholders will pass this as 'order.date' and 'shop.name' are in VALID_PLACEHOLDERS
        # but create_purchase_receipt will fail as it cannot provide data for them.
        create_purchase_receipt("ORD001", receipt_template_unresolvable_placeholder)
    except ValueError as e:
        print(f"\nError (expected for unresolvable valid placeholder): {e}")
