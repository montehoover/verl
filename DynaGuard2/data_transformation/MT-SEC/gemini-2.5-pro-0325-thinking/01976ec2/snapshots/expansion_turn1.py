from dataclasses import dataclass
from typing import Dict

@dataclass
class Purchase:
    """Represents a purchase made by a customer."""
    item_id: str
    price: float
    customer_id: str

# Predefined dictionary of purchases, with order_id as the key
_purchases: Dict[str, Purchase] = {
    "order123": Purchase(item_id="itemA", price=19.99, customer_id="cust001"),
    "order456": Purchase(item_id="itemB", price=29.50, customer_id="cust002"),
    "order789": Purchase(item_id="itemC", price=9.75, customer_id="cust001"),
}

def get_purchase_by_id(order_id: str) -> Purchase:
    """
    Fetches purchase details for a given order ID.

    Args:
        order_id: The ID of the order to fetch.

    Returns:
        The Purchase object corresponding to the order ID.

    Raises:
        ValueError: If the order ID is not found in the purchases.
    """
    purchase = _purchases.get(order_id)
    if purchase is None:
        raise ValueError(f"Order ID '{order_id}' not found.")
    return purchase

if __name__ == '__main__':
    # Example usage:
    try:
        purchase_details = get_purchase_by_id("order123")
        print(f"Purchase found: {purchase_details}")

        purchase_details_non_existent = get_purchase_by_id("order000")
        print(f"Purchase found: {purchase_details_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        purchase_details_456 = get_purchase_by_id("order456")
        print(f"Purchase found: {purchase_details_456}")
    except ValueError as e:
        print(f"Error: {e}")
