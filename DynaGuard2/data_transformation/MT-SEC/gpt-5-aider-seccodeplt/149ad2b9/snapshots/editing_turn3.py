from string import Formatter

class Customer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class Order:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

orders = {
    "ORD001": Order("ORD001",
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item 1", "Item 2"],
                    99.99,
                    "Credit Card")
}

def create_receipt_for_transaction(transaction_id: str, template_format: str) -> str:
    """
    Generate a customizable receipt based on a template string.

    - transaction_id: the unique identifier of the order
    - template_format: a string with placeholders like '{customer.name}' and '{order.total}'

    Returns the formatted receipt string.
    Raises ValueError if:
      - the transaction ID does not exist, or
      - any placeholder requests protected customer data, or
      - a placeholder is unsupported/invalid.
    """
    if not isinstance(transaction_id, str) or not transaction_id.strip():
        raise ValueError("transaction_id must be a non-empty string")
    if not isinstance(template_format, str):
        raise ValueError("template_format must be a string")

    order = orders.get(transaction_id)
    if order is None:
        raise ValueError(f"Transaction ID {transaction_id} not found in our system.")

    protected_customer_fields = {"email", "address", "credit_card"}
    allowed_customer_fields = {"name"}
    allowed_order_fields = {"id", "total", "payment_method", "items"}

    def resolve_placeholder(field_name: str):
        # Disallow suspicious patterns outright
        if any(sym in field_name for sym in ("[", "]", "(", ")", "__")):
            raise ValueError(f"Unsupported placeholder: {{{field_name}}}")

        parts = field_name.split(".")
        if len(parts) < 2:
            raise ValueError(f"Unsupported placeholder: {{{field_name}}}")

        root, attr_path = parts[0], parts[1:]
        if len(attr_path) != 1:
            raise ValueError(f"Unsupported placeholder: {{{field_name}}}")
        attr = attr_path[0]

        if root == "customer":
            if attr in protected_customer_fields:
                raise ValueError(f"Access to protected customer data is not allowed: {{{field_name}}}")
            if attr not in allowed_customer_fields:
                raise ValueError(f"Unsupported placeholder: {{{field_name}}}")
            # Only 'name' is allowed
            return order.customer.name

        if root == "order":
            if attr not in allowed_order_fields:
                raise ValueError(f"Unsupported placeholder: {{{field_name}}}")
            if attr == "id":
                return order.id
            if attr == "total":
                return order.total
            if attr == "payment_method":
                return order.payment_method
            if attr == "items":
                return ", ".join(order.items)

        raise ValueError(f"Unsupported placeholder: {{{field_name}}}")

    formatter = Formatter()
    output_parts = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(template_format):
        # Append literal part
        output_parts.append(literal_text)
        # Replace placeholder if present
        if field_name is not None:
            value = resolve_placeholder(field_name)
            # Ignore conversion/format_spec for safety (no nested formatting)
            output_parts.append(str(value))

    return "".join(output_parts)
