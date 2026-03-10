from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

@dataclass
class PurchaseOrder:
    order_id: str
    customer_id: str
    order_date: datetime
    items: List[Dict[str, any]]
    total_amount: float
    status: str

# Predefined dictionary of purchase orders
purchase_orders = {
    "PO-001": PurchaseOrder(
        order_id="PO-001",
        customer_id="CUST-123",
        order_date=datetime(2024, 1, 15, 10, 30),
        items=[
            {"product_id": "PROD-A", "quantity": 2, "price": 29.99},
            {"product_id": "PROD-B", "quantity": 1, "price": 49.99}
        ],
        total_amount=109.97,
        status="completed"
    ),
    "PO-002": PurchaseOrder(
        order_id="PO-002",
        customer_id="CUST-456",
        order_date=datetime(2024, 1, 16, 14, 45),
        items=[
            {"product_id": "PROD-C", "quantity": 3, "price": 19.99}
        ],
        total_amount=59.97,
        status="pending"
    ),
    "PO-003": PurchaseOrder(
        order_id="PO-003",
        customer_id="CUST-789",
        order_date=datetime(2024, 1, 17, 9, 15),
        items=[
            {"product_id": "PROD-A", "quantity": 1, "price": 29.99},
            {"product_id": "PROD-D", "quantity": 2, "price": 99.99}
        ],
        total_amount=229.97,
        status="shipped"
    )
}

def get_purchase_order(order_id: str) -> PurchaseOrder:
    if order_id not in purchase_orders:
        raise ValueError(f"Order ID '{order_id}' not found")
    return purchase_orders[order_id]
