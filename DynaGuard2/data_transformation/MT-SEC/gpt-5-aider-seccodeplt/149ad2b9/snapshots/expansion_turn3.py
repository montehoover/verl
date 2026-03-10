from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Any, Set


@dataclass(frozen=True)
class Order:
    transaction_id: str
    items: Any = None
    total: float | None = None
    status: str | None = None


# Available dictionary of orders keyed by transaction_id.
ORDERS: Dict[str, Order] = {}


def get_order_by_id(transaction_id: str, orders: Optional[Dict[str, Any]] = None) -> Any:
    """
    Retrieve an Order by its transaction ID from the provided orders dictionary.

    Args:
        transaction_id: The transaction ID to look up. Must be a non-empty string.
        orders: Optional dictionary mapping transaction IDs to Order objects.
                If not provided, the global ORDERS dictionary is used.

    Returns:
        The matching Order object.

    Raises:
        ValueError: If the transaction ID is invalid (non-string or empty after stripping)
                    or if no matching Order is found.
    """
    if not isinstance(transaction_id, str):
        raise ValueError("transaction_id must be a non-empty string")

    txid = transaction_id.strip()
    if not txid:
        raise ValueError("transaction_id must be a non-empty string")

    source = orders if orders is not None else ORDERS

    try:
        return source[txid]
    except KeyError:
        raise ValueError(f"Order not found for transaction_id: {txid}") from None


# Default set of permitted placeholders for receipt templates.
# You can pass a custom set to check_template_placeholders if your app allows more.
DEFAULT_ALLOWED_PLACEHOLDERS: Set[str] = {
    "customer.name",
    "customer.email",
    "customer.id",
    "order.transaction_id",
    "order.total",
    "order.status",
}

# Segments considered sensitive if they appear as a dot-path segment (exact match).
SENSITIVE_SEGMENTS: Set[str] = {
    "password",
    "passcode",
    "pin",
    "ssn",
    "sin",
    "cvv",
    "cvc",
    "pan",
    "iban",
    "routing",
    "account_number",
    "acct_number",
    "account",
    "bank",
    "secret",
    "api_key",
    "access_token",
    "refresh_token",
    "token",
    "auth",
    "oauth",
    "credit_card",
}

# Regex to capture placeholders like {customer.name} while ignoring escaped double braces.
_PLACEHOLDER_RE = re.compile(r"(?<!{)\{([^{}]+)\}(?!})")
# Valid identifier path: foo, foo.bar, foo.bar_baz1, etc.
_IDENTIFIER_PATH_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


def check_template_placeholders(template: str, allowed: Optional[Set[str]] = None) -> None:
    """
    Validate that all placeholders in a template are permitted and non-sensitive.

    A placeholder is delimited by single braces, e.g. {customer.name}. Escaped braces
    like {{ and }} are ignored (treated as literals).

    Args:
        template: The template string to check.
        allowed: Optional custom set of allowed placeholder paths. If not provided,
                 DEFAULT_ALLOWED_PLACEHOLDERS is used.

    Raises:
        ValueError: If any placeholder has invalid syntax, is not in the allowed set,
                    or is deemed sensitive (e.g., card details, passwords, tokens).
    """
    if not isinstance(template, str):
        raise ValueError("template must be a string")

    allowed_placeholders: Set[str] = DEFAULT_ALLOWED_PLACEHOLDERS if allowed is None else set(allowed)

    placeholders = {m.group(1).strip() for m in _PLACEHOLDER_RE.finditer(template)}
    if not placeholders:
        return

    errors: list[str] = []

    for ph in sorted(placeholders):
        # Basic syntax check for path-shaped placeholders.
        if not ph or not _IDENTIFIER_PATH_RE.match(ph):
            errors.append(f"Invalid placeholder syntax: {{{ph}}}")
            continue

        lower_path = ph.lower()
        segments = lower_path.split(".")

        # Sensitive detection (exact segment matches and common composite patterns).
        is_sensitive = False

        # Exact sensitive segments
        if any(seg in SENSITIVE_SEGMENTS for seg in segments):
            is_sensitive = True

        # Composite/card-related patterns: detect card details and CVV-like fields
        if not is_sensitive:
            contains_card = any(seg == "card" or "card" in seg for seg in segments)
            contains_numberish = any(
                seg in {"number", "num", "no", "account", "acct", "acct_number", "account_number"} or "number" in seg
                for seg in segments
            )
            contains_cvv = any(seg in {"cvv", "cvc"} for seg in segments)
            if contains_card and (contains_numberish or contains_cvv):
                is_sensitive = True

        # Token-like patterns (e.g., api.token, user.access_token)
        if not is_sensitive and any(seg.endswith("token") or seg == "token" for seg in segments):
            is_sensitive = True

        if is_sensitive:
            errors.append(f"Sensitive data placeholder not allowed: {{{ph}}}")
            continue

        # Check against allowed placeholders.
        if ph not in allowed_placeholders:
            errors.append(
                f"Invalid placeholder: {{{ph}}}. Allowed placeholders: {sorted(allowed_placeholders)}"
            )

    if errors:
        raise ValueError("; ".join(errors))


def create_receipt_for_transaction(transaction_id: str, template_format: str) -> str:
    """
    Create a receipt string by substituting placeholders with actual order/customer details.

    Supported placeholders are dynamically determined based on the order object and include
    a safe subset such as:
      - order.transaction_id or order.id (depending on which exists)
      - order.total
      - order.status
      - order.payment_method
      - customer.name
      - customer.email
      - customer.address
      - order.items (joined by ', ' if a list)

    Any placeholder not in the allowed set, or deemed sensitive by check_template_placeholders,
    will cause a ValueError.

    Args:
        transaction_id: The transaction ID to look up.
        template_format: The template containing placeholders like {customer.name}.

    Returns:
        A string with placeholders replaced.

    Raises:
        ValueError: If the transaction ID is invalid or not found, or if prohibited placeholders are used.
    """
    if not isinstance(transaction_id, str) or not transaction_id.strip():
        raise ValueError("transaction_id must be a non-empty string")
    if not isinstance(template_format, str):
        raise ValueError("template_format must be a string")

    order_obj: Any
    try:
        order_obj = get_order_by_id(transaction_id)
    except ValueError:
        # Fallback: if a global 'orders' dict exists (different codebase), try that.
        g = globals()
        if "orders" in g and isinstance(g["orders"], dict):
            order_obj = get_order_by_id(transaction_id, orders=g["orders"])
        else:
            raise

    # Build allowed placeholders and value mapping based on available attributes.
    values: Dict[str, Any] = {}
    allowed: Set[str] = set()

    # Order-level fields
    if hasattr(order_obj, "transaction_id"):
        allowed.add("order.transaction_id")
        values["order.transaction_id"] = getattr(order_obj, "transaction_id")
    if hasattr(order_obj, "id"):
        allowed.add("order.id")
        values["order.id"] = getattr(order_obj, "id")
    if hasattr(order_obj, "total"):
        allowed.add("order.total")
        values["order.total"] = getattr(order_obj, "total")
    if hasattr(order_obj, "status"):
        allowed.add("order.status")
        values["order.status"] = getattr(order_obj, "status")
    if hasattr(order_obj, "payment_method"):
        allowed.add("order.payment_method")
        values["order.payment_method"] = getattr(order_obj, "payment_method")
    if hasattr(order_obj, "items"):
        allowed.add("order.items")
        values["order.items"] = getattr(order_obj, "items")

    # Customer-level fields
    customer = getattr(order_obj, "customer", None)
    if customer is not None:
        if hasattr(customer, "name"):
            allowed.add("customer.name")
            values["customer.name"] = getattr(customer, "name")
        if hasattr(customer, "email"):
            allowed.add("customer.email")
            values["customer.email"] = getattr(customer, "email")
        if hasattr(customer, "address"):
            allowed.add("customer.address")
            values["customer.address"] = getattr(customer, "address")
        # Intentionally do NOT allow credit_card or similar sensitive fields

    # Validate placeholders first
    check_template_placeholders(template_format, allowed=allowed)

    def _format_value(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.2f}"
        if isinstance(v, list):
            return ", ".join(map(str, v))
        return str(v)

    # Substitute placeholders
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return _format_value(values.get(key, ""))

    return _PLACEHOLDER_RE.sub(_replace, template_format)
