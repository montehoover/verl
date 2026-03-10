import re

class Customer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class Order:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

orders = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item 1", "Item 2"],
                    99.99,
                    "Credit Card")
}

def extract_placeholders(template: str) -> list:
    """Extract all placeholders from the template string."""
    return re.findall(r'\{([^}]+)\}', template)

def validate_placeholders(placeholders: list) -> None:
    """Check if any placeholder tries to access restricted data."""
    restricted_fields = {'credit_card', 'creditcard', 'credit-card', 'cc'}
    
    for placeholder in placeholders:
        if any(restricted in placeholder.lower() for restricted in restricted_fields):
            raise ValueError(f"Access to restricted customer data is not allowed: {placeholder}")

def build_replacement_map(order: Order) -> dict:
    """Build a dictionary of valid replacements from the order data."""
    return {
        'order.id': order.id,
        'order.total': str(order.total),
        'order.payment_method': order.payment_method,
        'order.items': ', '.join(order.items),
        'customer.name': order.customer.name,
        'customer.email': order.customer.email,
        'customer.address': order.customer.address
    }

def get_attribute_value(order: Order, obj_name: str, attr_name: str) -> str:
    """Safely retrieve attribute value from order or customer object."""
    restricted_fields = {'credit_card', 'creditcard', 'credit-card', 'cc'}
    
    if obj_name == 'order' and hasattr(order, attr_name):
        return str(getattr(order, attr_name))
    elif obj_name == 'customer' and hasattr(order.customer, attr_name):
        if attr_name.lower() not in restricted_fields:
            return str(getattr(order.customer, attr_name))
    
    return None

def process_template(template: str, placeholders: list, replacements: dict, order: Order) -> str:
    """Process the template and replace all valid placeholders."""
    result = template
    
    for placeholder in placeholders:
        key = placeholder.strip()
        
        if key in replacements:
            result = result.replace(f'{{{placeholder}}}', replacements[key])
        else:
            parts = key.split('.')
            if len(parts) == 2:
                obj_name, attr_name = parts
                value = get_attribute_value(order, obj_name, attr_name)
                if value is not None:
                    result = result.replace(f'{{{placeholder}}}', value)
    
    return result

def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    # Check if order exists
    if purchase_id not in orders:
        raise ValueError(f"Order ID {purchase_id} does not exist")
    
    order = orders[purchase_id]
    
    # Extract and validate placeholders
    placeholders = extract_placeholders(receipt_template)
    validate_placeholders(placeholders)
    
    # Build replacement map
    replacements = build_replacement_map(order)
    
    # Process template
    return process_template(receipt_template, placeholders, replacements, order)
