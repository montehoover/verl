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


class SafeUser:
    """
    A restricted view of the User that blocks access to sensitive fields.
    """

    def __init__(self, user: User):
        self._user = user

    @property
    def name(self):
        return self._user.name

    @property
    def email(self):
        return self._user.email

    @property
    def address(self):
        return self._user.address

    @property
    def credit_card(self):
        # Access to sensitive information must raise ValueError
        raise ValueError("Attempted access to restricted customer data: credit_card")

    def __getattr__(self, item):
        # Any attribute not explicitly whitelisted is considered invalid
        raise AttributeError(f"'User' object has no attribute '{item}'")


class SafeOrder:
    """
    A simple proxy for Purchase to expose order fields to the template.
    """

    def __init__(self, purchase: Purchase):
        self._purchase = purchase

    @property
    def id(self):
        return self._purchase.id

    @property
    def total(self):
        return self._purchase.total

    @property
    def items(self):
        # Provide a readable representation
        return ", ".join(map(str, self._purchase.items))

    @property
    def payment_method(self):
        return self._purchase.payment_method

    def __getattr__(self, item):
        raise AttributeError(f"'Order' object has no attribute '{item}'")


def create_user_receipt(order_id: str, template: str) -> str:
    """
    Generate a receipt string for the given order ID using the provided template.

    Args:
        order_id: The order's unique identifier.
        template: A template string containing placeholders like '{customer.name}' and '{order.total}'.

    Returns:
        A formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid or if restricted customer data is accessed.
    """
    if not isinstance(order_id, str) or not order_id:
        raise ValueError("Invalid order ID")

    purchase = purchases.get(order_id)
    if purchase is None:
        raise ValueError("Invalid order ID")

    safe_user = SafeUser(purchase.client)
    safe_order = SafeOrder(purchase)

    # Use Python's format with named arguments. Accessing restricted fields will raise ValueError.
    try:
        result = template.format(customer=safe_user, order=safe_order)
    except ValueError:
        # Propagate restricted data access attempts as specified
        raise
    return result
