from string import Formatter
from typing import Tuple, Optional


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


def _first_customer_field(field_name: str) -> Optional[str]:
    """
    Given a field name extracted by string.Formatter, return the first attribute/key
    accessed on 'customer' if the field references the customer object, otherwise None.
    Examples:
      'customer' -> '' (empty string, meaning the object itself)
      'customer.name' -> 'name'
      'customer["email"]' -> 'email'
      'customer[address]' -> 'address'
      'customer.name.extra' -> 'name'
    """
    if not field_name.startswith('customer'):
        return None

    suffix = field_name[len('customer'):]
    if suffix == '':
        # Direct reference to the whole customer object
        return ''

    # If it's not attribute or indexing on 'customer', treat as not referencing customer
    if suffix[0] not in ('.', '['):
        return None

    # Attribute access: .field
    if suffix[0] == '.':
        # read until next '.' or '['
        i = 1
        token = []
        while i < len(suffix) and suffix[i] not in ('.', '['):
            token.append(suffix[i])
            i += 1
        return ''.join(token)

    # Indexing access: ['key'] or [key]
    # We try to capture a simple key possibly quoted.
    i = 1
    # Skip whitespace
    while i < len(suffix) and suffix[i].isspace():
        i += 1
    key = []
    if i < len(suffix) and suffix[i] in ("'", '"'):
        quote = suffix[i]
        i += 1
        while i < len(suffix) and suffix[i] != quote:
            key.append(suffix[i])
            i += 1
        # Move past closing quote if present
        if i < len(suffix) and suffix[i] == quote:
            i += 1
        # Move to closing bracket
        while i < len(suffix) and suffix[i] != ']':
            i += 1
    else:
        # Unquoted key: read until ']' or delimiter
        while i < len(suffix) and suffix[i] not in (']', '.', '['):
            key.append(suffix[i])
            i += 1
        # Move to closing bracket
        while i < len(suffix) and suffix[i] != ']':
            i += 1

    return ''.join(key).strip()


def create_receipt_for_transaction(transaction_id: str, template_format: str) -> str:
    """
    Generate a receipt string by substituting placeholders in template_format with real data
    from the order and customer.

    Arguments:
      - transaction_id: str, unique identifier of the order
      - template_format: str, the format template. Supports placeholders like:
            {order.id}, {order.total:.2f}, {customer.name}

    Returns:
      - A formatted receipt string.

    Raises:
      - ValueError if the order does not exist.
      - ValueError if the template requests restricted customer data (email, address, credit_card,
        or the entire customer object).
      - ValueError if the template contains unknown or invalid placeholders.
    """
    # Validate order existence
    try:
        order = orders[transaction_id]
    except KeyError:
        raise ValueError("Order ID does not exist")

    # Validate template placeholders to prevent access to restricted customer fields
    restricted_customer_fields = {'email', 'address', 'credit_card'}
    formatter = Formatter()

    for _, field_name, format_spec, conversion in formatter.parse(template_format):
        if not field_name:
            continue

        # str.format supports nested fields like 'order.total' and 'customer.name'
        # Identify any access to customer.* and validate allowed fields.
        first_customer_field = _first_customer_field(field_name)
        if first_customer_field is not None:
            # Disallow referencing the entire customer object or any non-allowed fields
            if first_customer_field == '' or first_customer_field in restricted_customer_fields:
                raise ValueError("Access to requested customer field is not allowed")
            # Only 'name' is permitted; anything else under customer.* is not allowed
            if first_customer_field != 'name':
                raise ValueError("Access to requested customer field is not allowed")

    # Perform formatting
    try:
        result = template_format.format(order=order, customer=order.customer)
    except (KeyError, AttributeError, IndexError, ValueError) as e:
        # Normalize any formatting errors into a ValueError per the spec
        raise ValueError(f"Invalid template placeholders: {e}") from e

    return result
