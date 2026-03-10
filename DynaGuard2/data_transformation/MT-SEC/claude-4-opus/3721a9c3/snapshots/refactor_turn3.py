import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='receipt_generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

def get_order_data(order_id: str) -> PurchaseOrder:
    """Retrieve order data for the given order ID."""
    if order_id not in purchase_orders:
        raise ValueError(f"Invalid order ID: {order_id}")
    return purchase_orders[order_id]

def validate_template(template: str) -> None:
    """Check template for restricted data access attempts."""
    if 'credit_card' in template:
        raise ValueError("Access to restricted customer data is not allowed")

def prepare_format_dictionary(order: PurchaseOrder) -> dict:
    """Create a dictionary mapping placeholder names to their values."""
    return {
        'customer.name': order.customer.name,
        'customer.email': order.customer.email,
        'customer.address': order.customer.address,
        'order.id': order.id,
        'order.total': order.total,
        'order.payment_method': order.payment_method,
        'order.items': ', '.join(order.items)
    }

def format_receipt(template: str, format_dict: dict) -> str:
    """Replace placeholders in template with actual values."""
    result = template
    for key, value in format_dict.items():
        placeholder = '{' + key + '}'
        result = result.replace(placeholder, str(value))
    return result

def generate_buyer_receipt(order_id: str, template: str) -> str:
    order = get_order_data(order_id)
    validate_template(template)
    format_dict = prepare_format_dictionary(order)
    
    # Log receipt generation
    logging.info(f"Receipt generated - Order ID: {order_id}, Customer: {order.customer.name}, Total: ${order.total:.2f}")
    
    return format_receipt(template, format_dict)
