import re
import logging
import os
from typing import Any, Tuple

# Configure logger for receipt generation activities
logger = logging.getLogger("receipt_generation")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), "receipt_generation.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False


def get_order_and_customer(order_id: str, store: dict) -> Tuple[Any, Any]:
    """
    Pure function to retrieve an order and its associated customer from a store.

    Args:
        order_id: The unique identifier for the order.
        store: A mapping of order_id to PurchaseOrder-like objects.

    Returns:
        A tuple of (order, customer).

    Raises:
        ValueError: If the order_id is invalid or the order does not exist.
    """
    if not isinstance(order_id, str) or not order_id:
        raise ValueError("Invalid order ID")

    if order_id not in store:
        raise ValueError("Invalid order ID")

    order = store[order_id]
    customer = getattr(order, "customer", None)
    return order, customer


def format_receipt(template: str, order: Any, customer: Any) -> str:
    """
    Pure function to format a receipt by interpolating placeholders with order/customer data.

    Placeholders:
        - {order.*}
        - {customer.*}
    Supports nested attribute access (dot notation) and list indexing via numeric segments.

    Args:
        template: The template string with placeholders.
        order: The order object.
        customer: The customer object.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If invalid placeholders are used or restricted data is accessed.
    """
    # Allowed placeholder roots
    roots = {
        "order": order,
        "customer": customer,
    }

    # Restricted customer fields that must not be exposed
    restricted_customer_attrs = {"credit_card"}

    # Regex to find simple {placeholder} tokens (no nested braces)
    token_pattern = re.compile(r"\{([^{}]+)\}")

    def resolve_token(token: str) -> str:
        token = token.strip()
        if not token:
            return ""

        parts = token.split(".")
        root_name = parts[0]
        if root_name not in roots:
            raise ValueError(f"Invalid placeholder root '{root_name}' in '{{{token}}}'")

        current: Any = roots[root_name]

        for i, attr in enumerate(parts[1:], start=1):
            if not attr:
                raise ValueError(f"Invalid placeholder '{{{token}}}'")

            # Disallow dunder or magic attribute access
            if attr.startswith("__") and attr.endswith("__"):
                raise ValueError(f"Restricted attribute access in '{{{token}}}'")

            # Prevent accessing restricted customer fields regardless of traversal path
            if attr in restricted_customer_attrs:
                raise ValueError(f"Attempt to access restricted customer data: '{attr}'")

            # Support list/tuple indexing if the segment is an integer
            index_used = False
            if isinstance(current, (list, tuple)) and attr.isdigit():
                idx = int(attr)
                try:
                    current = current[idx]
                    index_used = True
                except IndexError:
                    raise ValueError(f"Index out of range in '{{{token}}}'")

            if index_used:
                continue

            # Normal attribute access
            if hasattr(current, attr):
                current = getattr(current, attr)
                continue

            # Dict key access as a fallback
            if isinstance(current, dict) and attr in current:
                current = current[attr]
                continue

            raise ValueError(f"Unknown attribute '{attr}' in placeholder '{{{token}}}'")

        # Convert lists to a reader-friendly string
        if isinstance(current, list):
            return ", ".join(str(x) for x in current)
        return str(current)

    # Replace all placeholders in the template
    def replacement(match: re.Match) -> str:
        token_content = match.group(1)
        return resolve_token(token_content)

    return token_pattern.sub(replacement, template)


def generate_buyer_receipt(order_id: str, template: str) -> str:
    """
    Generate a receipt string for a given order by filling a template with order/customer data.

    Args:
        order_id: The unique identifier of the order to render.
        template: A template string containing placeholders such as '{customer.name}' or '{order.total}'.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid/unavailable or if restricted customer data is accessed.
    """
    # Access the provided global purchase_orders store
    try:
        store = purchase_orders  # type: ignore[name-defined]
    except NameError:
        raise ValueError("Order store is unavailable")

    order, customer = get_order_and_customer(order_id, store)
    receipt = format_receipt(template, order, customer)

    # Log receipt generation with order ID, customer name, and total amount
    try:
        customer_name = getattr(customer, "name", "<unknown>")
        total = getattr(order, "total", "<unknown>")
        logger.info(f"Generated receipt | order_id={order_id} | customer={customer_name} | total={total}")
    except Exception:
        # Ensure logging failures don't interrupt the primary flow
        pass

    return receipt
