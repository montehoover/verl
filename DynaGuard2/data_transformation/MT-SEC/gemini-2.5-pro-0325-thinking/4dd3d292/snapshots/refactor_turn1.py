import re

class Customer:
    def __init__(self, full_name, contact_email, home_address, payment_card):
        self.full_name = full_name
        self.contact_email = contact_email
        self.home_address = home_address
        self.payment_card = payment_card

class Order:
    def __init__(self, order_number, buyer, items_list, price_total, method_of_payment):
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
                    "Credit Card")
}

def create_purchase_receipt(order_id: str, receipt_template: str) -> str:
    """
    Generates a receipt for a given order ID using a template.

    Args:
        order_id: The identifier of the order.
        receipt_template: The template string for the receipt.

    Returns:
        A receipt string generated using the provided template.

    Raises:
        ValueError: If an invalid order ID is provided or if attempts are made
                    to access protected customer information in the template.
    """
    if order_id not in purchases:
        raise ValueError(f"Invalid order ID: {order_id}")

    order_obj = purchases[order_id]
    customer_obj = order_obj.buyer

    # Define protected attributes (content inside braces)
    protected_attributes = [
        'customer.home_address',
        'customer.payment_card'
    ]

    # Find all placeholders in the template
    # Placeholders are of the form {object.attribute}
    # This regex extracts 'object.attribute' from '{object.attribute}'
    placeholders_in_template = re.findall(r'\{([^}]+)\}', receipt_template)

    for placeholder_key in placeholders_in_template:
        if placeholder_key in protected_attributes:
            raise ValueError(
                f"Attempt to access protected customer information: {placeholder_key}"
            )

    # Build the receipt string by replacing placeholders
    receipt = receipt_template

    # Allowed placeholders and their corresponding values
    # Note: '{customer.name}' maps to customer_obj.full_name
    #       '{order.total}' maps to order_obj.price_total
    replacements = {
        '{customer.name}': customer_obj.full_name,
        '{customer.contact_email}': customer_obj.contact_email,
        '{order.order_number}': order_obj.order_number,
        '{order.items_list}': ", ".join(order_obj.items_list),
        '{order.total}': str(order_obj.price_total),
        '{order.method_of_payment}': order_obj.method_of_payment,
    }

    for placeholder, value in replacements.items():
        receipt = receipt.replace(placeholder, value)

    return receipt
