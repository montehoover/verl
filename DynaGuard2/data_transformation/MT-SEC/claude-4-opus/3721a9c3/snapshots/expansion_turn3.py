from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import re

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

def check_template_placeholders(template: str) -> bool:
    # Define valid placeholders
    valid_placeholders = {
        "{customer.name}",
        "{customer.id}",
        "{customer.email}",
        "{customer.phone}",
        "{customer.address}",
        "{order.id}",
        "{order.date}",
        "{order.total}",
        "{order.status}",
        "{order.items}",
        "{order.item_count}",
        "{company.name}",
        "{company.address}",
        "{company.phone}",
        "{company.email}",
        "{date}",
        "{time}",
        "{datetime}"
    }
    
    # Define restricted placeholders
    restricted_placeholders = {
        "{system.path}",
        "{system.env}",
        "{system.exec}",
        "{admin.token}",
        "{admin.password}",
        "{db.connection}",
        "{db.query}",
        "{api.key}",
        "{secret.key}"
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*\}'
    found_placeholders = set(re.findall(placeholder_pattern, template))
    
    # Check for restricted placeholders
    for placeholder in found_placeholders:
        if placeholder in restricted_placeholders:
            raise ValueError(f"Restricted placeholder '{placeholder}' is not allowed")
    
    # Check if all placeholders are valid
    for placeholder in found_placeholders:
        if placeholder not in valid_placeholders:
            raise ValueError(f"Invalid placeholder '{placeholder}'")
    
    return True

class Buyer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class PurchaseOrder:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

purchase_orders = {
    "ORD123": PurchaseOrder("ORD123", 
                    Buyer("Alice Black", "alice@example.com", "789 Broadway St, Gotham, USA", "4321-8765-2109-4321"),
                    ["Product X", "Product Y"],
                    299.50,
                    "Debit Card")
}

def generate_buyer_receipt(order_id: str, template: str) -> str:
    # Check for restricted placeholders
    restricted_placeholders = {
        "{system.path}",
        "{system.env}",
        "{system.exec}",
        "{admin.token}",
        "{admin.password}",
        "{db.connection}",
        "{db.query}",
        "{api.key}",
        "{secret.key}",
        "{customer.credit_card}"
    }
    
    placeholder_pattern = r'\{[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*\}'
    found_placeholders = set(re.findall(placeholder_pattern, template))
    
    for placeholder in found_placeholders:
        if placeholder in restricted_placeholders:
            raise ValueError(f"Restricted placeholder '{placeholder}' is not allowed")
    
    # Check if order ID exists
    if order_id not in purchase_orders:
        raise ValueError(f"Order ID '{order_id}' not found")
    
    # Get the order
    order = purchase_orders[order_id]
    
    # Prepare replacement values
    replacements = {
        "{customer.name}": order.customer.name,
        "{customer.email}": order.customer.email,
        "{customer.address}": order.customer.address,
        "{order.id}": order.id,
        "{order.total}": f"{order.total:.2f}",
        "{order.items}": ", ".join(order.items),
        "{order.item_count}": str(len(order.items)),
        "{order.payment_method}": order.payment_method,
        "{date}": datetime.now().strftime("%Y-%m-%d"),
        "{time}": datetime.now().strftime("%H:%M:%S"),
        "{datetime}": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Replace placeholders in template
    receipt = template
    for placeholder, value in replacements.items():
        receipt = receipt.replace(placeholder, value)
    
    return receipt
