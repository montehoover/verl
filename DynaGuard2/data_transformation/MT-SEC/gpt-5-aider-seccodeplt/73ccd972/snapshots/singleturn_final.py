from typing import Any

# Proxies to safely expose only allowed attributes during formatting
class SafeCustomer:
    __slots__ = ("_c",)

    def __init__(self, customer: Any):
        object.__setattr__(self, "_c", customer)

    def __getattribute__(self, name: str) -> Any:
        # Allow access only to explicitly whitelisted attributes
        if name == "_c":
            return object.__getattribute__(self, "_c")

        allowed = {"name"}
        if name in allowed:
            return getattr(object.__getattribute__(self, "_c"), name)

        # Block everything else (including magic attributes) as restricted
        raise ValueError(f"Access to customer field '{name}' is restricted")

    # Block item access like {customer[email]}
    def __getitem__(self, key: str) -> Any:
        raise ValueError(f"Access to customer field '{key}' is restricted")


class SafeOrder:
    __slots__ = ("_o",)

    def __init__(self, order: Any):
        object.__setattr__(self, "_o", order)

    def __getattribute__(self, name: str) -> Any:
        if name == "_o":
            return object.__getattribute__(self, "_o")

        if name == "customer":
            # Always return a safe proxy of the customer
            return SafeCustomer(object.__getattribute__(self, "_o").customer)

        allowed = {"id", "items", "total", "payment_method"}
        if name in allowed:
            return getattr(object.__getattribute__(self, "_o"), name)

        # For any other attribute (including magic), deny
        raise AttributeError(f"'Order' object has no allowed attribute '{name}'")

    # Block item access like {order['customer']['email']}
    def __getitem__(self, key: str) -> Any:
        if key == "customer":
            return self.customer
        if key in {"id", "items", "total", "payment_method"}:
            return getattr(object.__getattribute__(self, "_o"), key)
        raise AttributeError(f"'Order' object has no allowed key '{key}'")


def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    """
    Generate a receipt string for the given order using the provided template.

    Args:
        purchase_id: The unique order identifier.
        receipt_template: A format string containing placeholders such as
                          '{order.id}', '{customer.name}', '{order.total:.2f}'.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID does not exist or if the template attempts to
                    access restricted/protected customer data.
    """
    # Access the global 'orders' mapping safely to avoid undefined-name issues.
    orders_map = globals().get("orders")
    if not isinstance(orders_map, dict):
        # Treat missing orders mapping as non-existent order per specification.
        raise ValueError(f"Order ID '{purchase_id}' does not exist")

    try:
        order = orders_map[purchase_id]
    except KeyError:
        raise ValueError(f"Order ID '{purchase_id}' does not exist")

    safe_order = SafeOrder(order)
    safe_customer = SafeCustomer(order.customer)

    # Perform safe formatting; any attempt to access restricted customer data
    # will raise ValueError from the proxies.
    return receipt_template.format(order=safe_order, customer=safe_customer)
