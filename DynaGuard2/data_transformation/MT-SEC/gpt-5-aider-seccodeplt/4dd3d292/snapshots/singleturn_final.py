from string import Formatter
from types import SimpleNamespace


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
    "ORD001": Order(
        "ORD001",
        Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
        ["Item A", "Item B"],
        99.99,
        "Credit Card",
    )
}


def create_purchase_receipt(order_id: str, receipt_template: str) -> str:
    """
    Generate a purchase receipt by filling a template with order/customer data.

    Args:
        order_id: The identifier of the order to render.
        receipt_template: A template string using placeholders such as:
                          {order.order_number}, {order.price_total:.2f},
                          {customer.full_name}, {customer.contact_email}

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid, the template is invalid, or the
                    template attempts to access protected customer information.
    """
    if not isinstance(order_id, str) or not order_id.strip():
        raise ValueError("Invalid order ID.")

    if order_id not in purchases:
        raise ValueError("Invalid order ID.")

    order = purchases[order_id]
    customer = order.buyer

    # Define allowed attributes and protected fields
    protected_customer_fields = {"home_address", "payment_card"}
    allowed_customer_fields = {"full_name", "contact_email", "name", "email"}
    allowed_order_fields = {
        "order_number",
        "items_list",
        "price_total",
        "method_of_payment",
        # friendly aliases
        "id",
        "items",
        "total",
        "payment_method",
    }

    # Pre-validate template fields to ensure only allowed data is accessed
    formatter = Formatter()
    for _, field_name, format_spec, conversion in formatter.parse(receipt_template):
        if field_name is None:
            continue

        # Disallow indexing and call syntax
        if any(ch in field_name for ch in "[]()"):
            raise ValueError("Invalid placeholder syntax in template.")

        parts = field_name.split(".")
        if len(parts) == 0:
            continue

        root = parts[0]
        if root not in ("order", "customer"):
            raise ValueError("Invalid placeholder root in template. Use 'order' or 'customer'.")

        if len(parts) == 1:
            # Disallow bare object rendering to avoid leaking representations
            raise ValueError("Placeholders must reference a specific attribute (e.g., {order.order_number}).")

        if len(parts) > 2:
            raise ValueError("Nested attributes are not allowed in template.")

        attr = parts[1]
        if attr.startswith("_"):
            raise ValueError("Access to protected information is not allowed.")

        if root == "customer":
            if attr in protected_customer_fields:
                raise ValueError("Access to protected customer information is not allowed.")
            if attr not in allowed_customer_fields:
                raise ValueError(f"Invalid customer attribute '{attr}' in template.")
        else:  # root == "order"
            if attr not in allowed_order_fields:
                raise ValueError(f"Invalid order attribute '{attr}' in template.")

        # Disallow nested fields inside format spec
        if format_spec and ("{" in format_spec or "}" in format_spec):
            raise ValueError("Invalid format specifier in template.")

        # Only allow default string conversion (or none)
        if conversion not in (None, "s"):
            raise ValueError("Invalid conversion in template.")

    # Build safe proxies exposing only allowed fields (plus friendly aliases)
    order_proxy = SimpleNamespace(
        order_number=order.order_number,
        items_list=order.items_list,
        price_total=order.price_total,
        method_of_payment=order.method_of_payment,
        # aliases
        id=order.order_number,
        items=order.items_list,
        total=order.price_total,
        payment_method=order.method_of_payment,
    )
    customer_proxy = SimpleNamespace(
        full_name=customer.full_name,
        contact_email=customer.contact_email,
        # aliases
        name=customer.full_name,
        email=customer.contact_email,
    )

    try:
        rendered = formatter.vformat(
            receipt_template,
            args=(),
            kwargs={"order": order_proxy, "customer": customer_proxy},
        )
    except Exception as exc:
        # Convert any formatting error into a consistent ValueError for callers
        raise ValueError(f"Failed to generate receipt: {exc}") from exc

    return rendered
