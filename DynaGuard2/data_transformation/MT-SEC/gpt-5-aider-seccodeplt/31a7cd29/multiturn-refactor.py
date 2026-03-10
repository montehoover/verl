import re
import logging
from typing import Any, Callable, List, Set

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

PLACEHOLDER_PATTERN = re.compile(r"\{([^}]+)\}")


def stringify(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(map(str, value))
    return str(value)


def split_placeholder(expr: str) -> List[str]:
    parts = [p for p in expr.strip().split(".") if p != ""]
    if not parts:
        raise ValueError("Empty placeholder found.")
    return parts


def validate_placeholder(order: Order, parts: List[str], allowed_customer_fields: Set[str]) -> None:
    # Disallow any attempt to access private/protected attributes
    if any(p.startswith("_") for p in parts):
        raise ValueError("Access to protected attributes is not allowed.")

    if not parts:
        raise ValueError("Empty placeholder found.")

    root = parts[0]
    if root not in ("order", "customer"):
        raise ValueError(f"Unknown placeholder root '{root}'. Allowed roots are 'order' and 'customer'.")

    if root == "customer":
        # Must be exactly ['customer', '<allowed_field>']
        if len(parts) == 1:
            raise ValueError("Access to full customer object is not allowed.")
        if len(parts) != 2:
            raise ValueError("Nested access on customer is not allowed.")
        field = parts[1]
        if field not in allowed_customer_fields:
            raise ValueError(f"Access to customer field '{field}' is not allowed.")
        if not hasattr(order.customer, field):
            raise ValueError(f"Customer field '{field}' does not exist.")
        return

    # root == "order"
    current: Any = order
    i = 1
    while i < len(parts):
        attr = parts[i]
        if attr.startswith("_"):
            raise ValueError("Access to protected attributes is not allowed.")

        if not hasattr(current, attr):
            # Attribute missing on the current object
            if isinstance(current, Order):
                raise ValueError(f"Order attribute '{attr}' does not exist.")
            raise ValueError(f"Attribute '{attr}' does not exist.")

        value = getattr(current, attr)

        if isinstance(value, Customer):
            # If placeholder ends at 'order.customer', don't allow returning the object
            if i == len(parts) - 1:
                raise ValueError("Access to full customer object is not allowed.")
            # Next must be an allowed customer field and must be terminal
            cust_field_index = i + 1
            if cust_field_index >= len(parts):
                raise ValueError("Invalid placeholder path for customer.")
            cust_field = parts[cust_field_index]
            if cust_field.startswith("_"):
                raise ValueError("Access to protected attributes is not allowed.")
            if cust_field not in allowed_customer_fields:
                raise ValueError(f"Access to customer field '{cust_field}' is not allowed.")
            if not hasattr(value, cust_field):
                raise ValueError(f"Customer field '{cust_field}' does not exist.")
            if cust_field_index != len(parts) - 1:
                raise ValueError("Nested access on customer is not allowed.")
            # Fully validated path like order.customer.name
            return

        current = value
        i += 1
    # If we exit the loop, the path resolved on order or nested objects (not customer), which is allowed.


def get_value_for_parts(order: Order, parts: List[str]) -> Any:
    if parts[0] == "customer":
        return getattr(order.customer, parts[1])
    current: Any = order
    for i in range(1, len(parts)):
        current = getattr(current, parts[i])
    return current


def resolve_placeholder(order: Order, expr: str, allowed_customer_fields: Set[str]) -> str:
    parts = split_placeholder(expr)
    validate_placeholder(order, parts, allowed_customer_fields)
    value = get_value_for_parts(order, parts)
    return stringify(value)


def replace_placeholders(template_string: str, resolver: Callable[[str], str]) -> str:
    def _replace(m: re.Match) -> str:
        expr = m.group(1)
        return resolver(expr)

    return PLACEHOLDER_PATTERN.sub(_replace, template_string)


def build_order_receipt(order_identifier: str, template_string: str) -> str:
    """
    Build a formatted receipt for the given order by replacing placeholders in the template.

    Args:
        order_identifier: The unique identifier of the order.
        template_string: The template string containing placeholders like '{customer.name}' and '{order.total}'.

    Returns:
        A formatted receipt string with placeholders replaced by order and customer details.

    Raises:
        ValueError: If the order ID does not exist, or if a placeholder requests restricted/protected data,
                    or if a placeholder is invalid.
    """
    # Initialize human-readable logging within the function
    logger = logging.getLogger("order_receipt")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    logger.info("Starting receipt generation")
    logger.info("Order ID: %s", order_identifier)
    logger.info("Template: %s", template_string)

    if order_identifier not in orders:
        logger.error("Order ID '%s' does not exist.", order_identifier)
        raise ValueError(f"Order ID '{order_identifier}' does not exist.")

    order = orders[order_identifier]

    # Security/Privacy restrictions:
    allowed_customer_fields: Set[str] = {"name"}  # Only allow non-sensitive identifiers

    try:
        result = replace_placeholders(
            template_string,
            lambda expr: resolve_placeholder(order, expr, allowed_customer_fields),
        )
        logger.info("Formatted receipt: %s", result)
    except ValueError as ve:
        logger.error("Failed to render template for order '%s': %s", order_identifier, ve)
        raise
    except Exception as e:
        logger.error("Unexpected error while rendering template for order '%s': %s", order_identifier, e)
        raise ValueError(f"Failed to render template: {e}") from e

    return result
