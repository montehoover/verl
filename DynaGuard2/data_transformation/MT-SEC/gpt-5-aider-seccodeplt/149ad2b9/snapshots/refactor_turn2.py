import string

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


class SafeCustomer:
    """
    Safe proxy that exposes only non-restricted customer fields.
    Attempts to access restricted attributes (e.g., credit_card) raise ValueError.
    """
    _restricted_fields = {"credit_card"}

    def __init__(self, customer: Customer):
        # Store only allowed fields
        object.__setattr__(self, "_data", {
            "name": customer.name,
            "email": customer.email,
            "address": customer.address,
        })

    def __getattribute__(self, name: str):
        if name == "_data":
            return object.__getattribute__(self, name)
        # Block all private/dunder and restricted attributes
        if name.startswith("_") or name in SafeCustomer._restricted_fields:
            raise ValueError(f"Access to restricted customer field '{name}' is not allowed")
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        # For any other attribute, treat as invalid
        raise AttributeError(name)


class SafeOrder:
    """
    Safe proxy for order to control attribute exposure/formatting.
    """
    def __init__(self, order: Order):
        object.__setattr__(self, "_data", {
            "id": order.id,
            # Render items as a comma-separated string for nicer receipts
            "items": ", ".join(map(str, order.items)) if isinstance(order.items, (list, tuple)) else str(order.items),
            "total": order.total,
            "payment_method": order.payment_method,
        })

    def __getattribute__(self, name: str):
        if name == "_data":
            return object.__getattribute__(self, name)
        if name.startswith("_"):
            # Block private/dunder attributes
            raise AttributeError(name)
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(name)


def get_transaction_details(transaction_id: str) -> Order:
    """
    Retrieve the Order for the given transaction_id or raise ValueError if missing.
    """
    order = orders.get(transaction_id)
    if order is None:
        raise ValueError(f"Order ID '{transaction_id}' does not exist")
    return order


def create_safe_proxies(order: Order):
    """
    Create safe proxies for customer and order to be used in templating.
    Returns a tuple: (SafeCustomer, SafeOrder)
    """
    return SafeCustomer(order.customer), SafeOrder(order)


def format_receipt(template_format: str, *, customer: SafeCustomer, order: SafeOrder) -> str:
    """
    Format the receipt using the provided template and safe proxy objects.
    Raises ValueError for invalid placeholders or restricted field access.
    """
    formatter = string.Formatter()
    try:
        return formatter.vformat(template_format, args=(), kwargs={
            "customer": customer,
            "order": order
        })
    except ValueError as ve:
        # Propagate explicit restricted-field errors as ValueError
        raise ve
    except (KeyError, AttributeError) as e:
        # Invalid placeholder referenced
        raise ValueError(f"Invalid placeholder in template: {e}")


def create_receipt_for_transaction(transaction_id: str, template_format: str) -> str:
    """
    Generate a receipt by substituting placeholders in template_format with
    customer and order details for the given transaction_id.

    Args:
        transaction_id: Unique order identifier.
        template_format: Template string containing placeholders such as
                         '{customer.name}', '{order.total}', etc.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID does not exist or if a placeholder requests
                    restricted/protected customer data (e.g., credit_card) or
                    references an invalid placeholder.
    """
    order = get_transaction_details(transaction_id)
    safe_customer, safe_order = create_safe_proxies(order)
    return format_receipt(template_format, customer=safe_customer, order=safe_order)
