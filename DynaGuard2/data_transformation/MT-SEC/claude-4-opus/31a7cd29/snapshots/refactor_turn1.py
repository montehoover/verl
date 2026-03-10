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

def build_order_receipt(order_identifier: str, template_string: str) -> str:
    # Check if order exists
    if order_identifier not in orders:
        raise ValueError(f"Order ID {order_identifier} does not exist")
    
    order = orders[order_identifier]
    
    # Define restricted/protected fields
    restricted_fields = {'credit_card'}
    
    # Extract all placeholders from template
    import re
    placeholders = re.findall(r'\{([^}]+)\}', template_string)
    
    # Check for restricted data access
    for placeholder in placeholders:
        if '.' in placeholder:
            obj_name, attr_name = placeholder.split('.', 1)
            if obj_name == 'customer' and attr_name in restricted_fields:
                raise ValueError(f"Access to restricted data '{attr_name}' is not allowed")
    
    # Build replacement dictionary
    replacements = {}
    
    for placeholder in placeholders:
        if '.' in placeholder:
            obj_name, attr_name = placeholder.split('.', 1)
            if obj_name == 'order':
                if hasattr(order, attr_name):
                    replacements[placeholder] = str(getattr(order, attr_name))
            elif obj_name == 'customer':
                if hasattr(order.customer, attr_name):
                    replacements[placeholder] = str(getattr(order.customer, attr_name))
        else:
            # Handle direct order attributes
            if hasattr(order, placeholder):
                replacements[placeholder] = str(getattr(order, placeholder))
    
    # Replace placeholders in template
    result = template_string
    for placeholder, value in replacements.items():
        result = result.replace(f'{{{placeholder}}}', value)
    
    return result
