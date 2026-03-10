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

def build_order_receipt(order_identifier, template_string):
    if order_identifier not in orders:
        raise ValueError(f"Order ID '{order_identifier}' does not exist")
    
    order = orders[order_identifier]
    
    # List of restricted customer attributes
    restricted_attrs = ['credit_card', 'address']
    
    # Check for restricted data access in template
    for attr in restricted_attrs:
        if f'{{customer.{attr}}}' in template_string:
            raise ValueError(f"Access to customer.{attr} is restricted")
    
    # Create a safe replacement function
    def safe_replace(match):
        placeholder = match.group(1)
        
        # Handle order attributes
        if placeholder.startswith('order.'):
            attr = placeholder[6:]  # Remove 'order.' prefix
            if hasattr(order, attr):
                return str(getattr(order, attr))
        
        # Handle customer attributes
        elif placeholder.startswith('customer.'):
            attr = placeholder[9:]  # Remove 'customer.' prefix
            if attr in restricted_attrs:
                raise ValueError(f"Access to customer.{attr} is restricted")
            if hasattr(order.customer, attr):
                return str(getattr(order.customer, attr))
        
        # Return the original placeholder if not found
        return match.group(0)
    
    # Replace placeholders using regex
    import re
    formatted_string = re.sub(r'\{([^}]+)\}', safe_replace, template_string)
    
    return formatted_string
