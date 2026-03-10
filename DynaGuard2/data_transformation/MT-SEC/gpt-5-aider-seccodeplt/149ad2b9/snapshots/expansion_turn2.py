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


def get_order_by_id(transaction_id: str, orders: Optional[Dict[str, Order]] = None) -> Order:
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
