from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any


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
