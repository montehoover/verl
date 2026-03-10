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

def _get_order_details(order_id: str, purchases_db: dict) -> Order:
    """
    Retrieves order details for a given order ID.

    Args:
        order_id: The identifier of the order.
        purchases_db: A dictionary of purchases.

    Returns:
        The Order object.

    Raises:
        ValueError: If an invalid order ID is provided.
    """
    if order_id not in purchases_db:
        raise ValueError(f"Invalid order ID: {order_id}")
    return purchases_db[order_id]

def _format_receipt(order_obj: Order, customer_obj: Customer, receipt_template: str) -> str:
    """
    Formats the receipt string using order and customer details.

    Args:
        order_obj: The Order object.
        customer_obj: The Customer object.
        receipt_template: The template string for the receipt.

    Returns:
        A formatted receipt string.

    Raises:
        ValueError: If attempts are made to access protected customer information.
    """
    # Define protected attributes (content inside braces)
    protected_attributes = [
        'customer.home_address',
        'customer.payment_card'
    ]

    # Find all placeholders in the template
    placeholders_in_template = re.findall(r'\{([^}]+)\}', receipt_template)

    for placeholder_key in placeholders_in_template:
        if placeholder_key in protected_attributes:
            raise ValueError(
                f"Attempt to access protected customer information: {placeholder_key}"
            )

    # Allowed placeholders and their corresponding values
    replacements = {
        '{customer.name}': customer_obj.full_name,
        '{customer.contact_email}': customer_obj.contact_email,
        '{order.order_number}': order_obj.order_number,
        '{order.items_list}': ", ".join(order_obj.items_list),
        '{order.total}': str(order_obj.price_total),
        '{order.method_of_payment}': order_obj.method_of_payment,
    }

    receipt = receipt_template
    for placeholder, value in replacements.items():
        receipt = receipt.replace(placeholder, value)

    return receipt

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
    order_obj = _get_order_details(order_id, purchases)
    customer_obj = order_obj.buyer
    
    return _format_receipt(order_obj, customer_obj, receipt_template)
