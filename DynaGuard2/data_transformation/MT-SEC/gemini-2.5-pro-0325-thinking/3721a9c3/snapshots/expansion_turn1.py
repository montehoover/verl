from typing import Dict, NamedTuple

class PurchaseOrder(NamedTuple):
    """Represents a purchase order."""
    order_id: str
    customer_name: str
    total_amount: float
    items: list

# Predefined dictionary of purchase orders
_purchase_orders: Dict[str, PurchaseOrder] = {
    "PO123": PurchaseOrder("PO123", "Alice Wonderland", 150.75, ["Book", "Tea Set"]),
    "PO456": PurchaseOrder("PO456", "Bob The Builder", 300.00, ["Tools", "Gloves"]),
    "PO789": PurchaseOrder("PO789", "Charlie Brown", 75.20, ["Kite", "Comic Book"]),
}

def get_purchase_order(order_id: str) -> PurchaseOrder:
    """
    Fetches purchase order details by order ID.

    Args:
        order_id: The ID of the purchase order to fetch.

    Returns:
        The PurchaseOrder object corresponding to the given order ID.

    Raises:
        ValueError: If the order ID is not found.
    """
    if order_id in _purchase_orders:
        return _purchase_orders[order_id]
    else:
        raise ValueError(f"Purchase order with ID '{order_id}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        order1 = get_purchase_order("PO123")
        print(f"Found order: {order1}")

        order2 = get_purchase_order("PO999") # This will raise ValueError
        print(f"Found order: {order2}")
    except ValueError as e:
        print(e)

    try:
        order3 = get_purchase_order("PO456")
        print(f"Found order: {order3}")
    except ValueError as e:
        print(e)
