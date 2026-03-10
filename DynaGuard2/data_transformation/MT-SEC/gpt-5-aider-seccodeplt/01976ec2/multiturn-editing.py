import string


class User:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card


class Purchase:
    def __init__(self, id, client, items, total, payment_method):
        self.id = id
        self.client = client
        self.items = items
        self.total = total
        self.payment_method = payment_method


purchases = {
    "ORD002": Purchase(
        "ORD002",
        User(
            "John Smith",
            "john.smith@example.com",
            "456 Park Ave, Metropolis, USA",
            "9876-5432-1098-7654",
        ),
        ["Item A", "Item B"],
        150.50,
        "Credit Card",
    )
}


_RESTRICTED_CUSTOMER_FIELDS = {"email", "credit_card"}


class _PlaceholderChain:
    """
    A stand-in for unknown placeholders that keeps them unchanged after formatting.
    """

    def __init__(self, path: str):
        self.path = path

    def __getattr__(self, name: str):
        return _PlaceholderChain(f"{self.path}.{name}")

    def __getitem__(self, key):
        return _PlaceholderChain(f"{self.path}[{key!r}]")

    def __format__(self, format_spec: str) -> str:
        if format_spec:
            return "{" + f"{self.path}:{format_spec}" + "}"
        return "{" + self.path + "}"

    def __str__(self) -> str:
        return "{" + self.path + "}"


class _SafeReceiptFormatter(string.Formatter):
    """
    Formatter that:
    - Resolves attributes safely from provided context.
    - Leaves unknown placeholders unchanged.
    - Raises on attempts to access restricted customer fields.
    """

    def get_field(self, field_name, args, kwargs):
        # Block restricted customer fields
        if self._is_restricted_field(field_name):
            raise ValueError(f"Access to restricted customer data: {field_name}")

        # Resolve dotted attribute paths manually, leaving unknowns unchanged
        parts = field_name.split(".")
        root_key = parts[0]

        if root_key not in kwargs:
            return _PlaceholderChain(field_name), field_name

        current = kwargs[root_key]

        for part in parts[1:]:
            # Basic attribute resolution only; indexing not supported -> leave unchanged
            if "[" in part or "]" in part:
                return _PlaceholderChain(field_name), field_name
            try:
                current = getattr(current, part)
            except AttributeError:
                return _PlaceholderChain(field_name), field_name

        return current, field_name

    @staticmethod
    def _is_restricted_field(field_name: str) -> bool:
        parts = field_name.split(".")
        if not parts:
            return False

        # Direct "customer.<field>"
        if parts[0] == "customer" and len(parts) >= 2 and parts[1] in _RESTRICTED_CUSTOMER_FIELDS:
            return True

        # "order.client.<field>"
        if (
            parts[0] == "order"
            and len(parts) >= 3
            and parts[1] == "client"
            and parts[2] in _RESTRICTED_CUSTOMER_FIELDS
        ):
            return True

        return False


def create_user_receipt(order_id: str, template: str) -> str:
    """
    Create a user-facing receipt string by formatting the provided template with data
    from the specified order.

    Placeholders:
      - {order.id}, {order.total}, {order.payment_method}, {order.items}, etc.
      - {customer.name} (allowed)
      - Access to restricted customer fields like {customer.email} or
        {customer.credit_card} will raise ValueError.

    Args:
        order_id (str): The ID of the order to format.
        template (str): The template string containing placeholders.

    Returns:
        str: The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid or if restricted customer data is accessed.
    """
    if not isinstance(order_id, str):
        raise ValueError("order_id must be a string")
    if not isinstance(template, str):
        raise ValueError("template must be a string")

    order = purchases.get(order_id)
    if not order:
        raise ValueError(f"Invalid order ID: {order_id}")

    context = {
        "order": order,
        "customer": order.client,
    }

    formatter = _SafeReceiptFormatter()
    return formatter.format(template, **context)
