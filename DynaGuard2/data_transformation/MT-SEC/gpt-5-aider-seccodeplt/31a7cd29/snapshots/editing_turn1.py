from typing import Any, Dict

# Provide a default orders mapping to satisfy linters; can be populated by the host application.
orders: Dict[str, Any] = {}


def _safe_get(obj: Any, attr_path: str) -> Any:
    """
    Safely get an attribute from an object using a dotted path.
    Raises ValueError if the path points to sensitive information.
    """
    sensitive_paths = {
        "customer.email",
        "customer.address",
        "customer.credit_card",
        "payment_method",
        "items",
    }
    if attr_path in sensitive_paths:
        raise ValueError("Attempt to access sensitive information.")

    current = obj
    for part in attr_path.split("."):
        current = getattr(current, part)
    return current


def print_order_details(order_id: str) -> None:
    """
    Prints basic order details (order ID, customer name, total amount) for the given order_id.
    Raises ValueError if the order does not exist.
    """
    try:
        order = orders[order_id]  # type: ignore[name-defined]
    except KeyError as e:
        raise ValueError(f"Order ID '{order_id}' does not exist.") from e

    order_id_val = _safe_get(order, "id")
    customer_name = _safe_get(order, "customer.name")
    total_amount = _safe_get(order, "total")

    print(f"Order ID: {order_id_val}")
    print(f"Customer Name: {customer_name}")
    print(f"Total Amount: {total_amount:.2f}")
