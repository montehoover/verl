from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PurchaseOrderItem:
    sku: str
    quantity: int
    unit_price: float


@dataclass(frozen=True)
class PurchaseOrder:
    order_id: str
    customer_id: str
    items: List[PurchaseOrderItem]
    currency: str = "USD"


# Predefined purchase orders dictionary
PURCHASE_ORDERS: Dict[str, PurchaseOrder] = {
    "PO-1001": PurchaseOrder(
        order_id="PO-1001",
        customer_id="CUST-001",
        items=[
            PurchaseOrderItem(sku="SKU-ABC", quantity=2, unit_price=49.99),
            PurchaseOrderItem(sku="SKU-XYZ", quantity=1, unit_price=199.00),
        ],
    ),
    "PO-1002": PurchaseOrder(
        order_id="PO-1002",
        customer_id="CUST-002",
        items=[
            PurchaseOrderItem(sku="SKU-DEF", quantity=3, unit_price=15.50),
        ],
        currency="EUR",
    ),
    "PO-1003": PurchaseOrder(
        order_id="PO-1003",
        customer_id="CUST-003",
        items=[
            PurchaseOrderItem(sku="SKU-GHI", quantity=5, unit_price=9.99),
            PurchaseOrderItem(sku="SKU-JKL", quantity=2, unit_price=24.75),
        ],
    ),
}


def get_purchase_order(order_id: str) -> PurchaseOrder:
    """
    Fetch a PurchaseOrder by its ID.

    Args:
        order_id: The identifier of the purchase order.

    Returns:
        The corresponding PurchaseOrder object.

    Raises:
        ValueError: If the order ID is not found.
    """
    try:
        po = PURCHASE_ORDERS[order_id]
    except KeyError:
        raise ValueError(f"Purchase order not found for ID: {order_id}")
    return po
