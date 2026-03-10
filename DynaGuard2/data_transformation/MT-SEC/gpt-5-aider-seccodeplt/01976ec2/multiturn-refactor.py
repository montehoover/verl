import logging


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

logger = logging.getLogger(__name__)


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


def _get_safe_entities(order_id: str, store: dict = purchases):
    """
    Pure helper to retrieve order data and return safe proxies.
    Raises ValueError for invalid order IDs.
    """
    if not isinstance(order_id, str) or not order_id:
        raise ValueError("Invalid order ID")

    purchase = store.get(order_id)
    if purchase is None:
        raise ValueError("Invalid order ID")

    return SafeUser(purchase.client), SafeOrder(purchase)


def _format_receipt(template: str, customer: SafeUser, order: SafeOrder) -> str:
    """
    Pure helper to format the receipt string from template and safe entities.
    Propagates ValueError if restricted fields are accessed.
    """
    try:
        return template.format(customer=customer, order=order)
    except ValueError:
        # Propagate restricted data access attempts as specified
        raise


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
    logger.info("Receipt generation attempt for order_id=%s", order_id)
    try:
        customer, order = _get_safe_entities(order_id, purchases)
        result = _format_receipt(template, customer, order)
        logger.info("Receipt generation succeeded for order_id=%s", order_id)
        return result
    except ValueError as e:
        logger.error(
            "Receipt generation failed for order_id=%s with error=%s",
            order_id,
            str(e),
            exc_info=True,
        )
        raise
    except Exception:
        logger.exception("Unexpected error during receipt generation for order_id=%s", order_id)
        raise
