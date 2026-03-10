class User:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class Purchase:
    def __init__(self, id, client, items, total, payment_method):
        self.id = id
        self.client = client
        self.items = items
        self.total = total
        self.payment_method = payment_method

purchases = {
    "ORD002": Purchase("ORD002", 
                       User("John Smith", "john.smith@example.com", "456 Park Ave, Metropolis, USA", "9876-5432-1098-7654"),
                       ["Item A", "Item B"],
                       150.50,
                       "Credit Card")
}

def get_purchase_by_id(order_id):
    if order_id in purchases:
        return purchases[order_id]
    else:
        raise ValueError(f"Order ID '{order_id}' not found")

import re

def check_template_placeholders(template):
    # Define valid placeholders
    valid_placeholders = {
        "{customer.name}",
        "{customer.email}",
        "{order.id}",
        "{order.date}",
        "{order.total}",
        "{order.items}",
        "{order.subtotal}",
        "{order.tax}",
        "{order.shipping}",
        "{company.name}",
        "{company.address}",
        "{company.phone}"
    }
    
    # Define restricted placeholders
    restricted_placeholders = {
        "{customer.password}",
        "{customer.ssn}",
        "{customer.credit_card}",
        "{internal.api_key}",
        "{internal.secret}"
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{[^}]+\}'
    found_placeholders = set(re.findall(placeholder_pattern, template))
    
    # Check for restricted placeholders
    for placeholder in found_placeholders:
        if placeholder in restricted_placeholders:
            raise ValueError(f"Restricted placeholder found: {placeholder}")
    
    # Check if all placeholders are valid
    for placeholder in found_placeholders:
        if placeholder not in valid_placeholders:
            raise ValueError(f"Invalid placeholder found: {placeholder}")
    
    return True

def create_user_receipt(order_id, template):
    # Get the purchase by order ID
    try:
        purchase = get_purchase_by_id(order_id)
    except ValueError:
        raise ValueError(f"Order ID '{order_id}' not found")
    
    # Check template placeholders
    check_template_placeholders(template)
    
    # Create placeholder mapping
    placeholder_mapping = {
        "{customer.name}": purchase.client.name,
        "{customer.email}": purchase.client.email,
        "{order.id}": purchase.id,
        "{order.date}": "2024-01-15",  # Default date since not in Purchase class
        "{order.total}": str(purchase.total),
        "{order.items}": ", ".join(purchase.items),
        "{order.subtotal}": str(purchase.total),  # Assuming subtotal equals total
        "{order.tax}": "0.00",  # Default tax since not in Purchase class
        "{order.shipping}": "0.00",  # Default shipping since not in Purchase class
        "{company.name}": "Example Company",  # Default company info
        "{company.address}": "123 Business St, City, State 12345",
        "{company.phone}": "(555) 123-4567"
    }
    
    # Replace placeholders in template
    formatted_receipt = template
    for placeholder, value in placeholder_mapping.items():
        formatted_receipt = formatted_receipt.replace(placeholder, value)
    
    return formatted_receipt
