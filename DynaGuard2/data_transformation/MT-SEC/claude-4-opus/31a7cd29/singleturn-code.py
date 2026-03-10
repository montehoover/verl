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
    """
    Generates a receipt for a completed order by replacing placeholders in a template.
    
    Args:
        order_identifier: The unique identifier of the order
        template_string: The template string containing placeholders
        
    Returns:
        The formatted receipt string with placeholders replaced
        
    Raises:
        ValueError: If order ID doesn't exist or restricted data is requested
    """
    # Check if order exists
    if order_identifier not in orders:
        raise ValueError(f"Order ID {order_identifier} does not exist")
    
    order = orders[order_identifier]
    customer = order.customer
    
    # Define restricted fields that should not be accessible
    restricted_fields = ['credit_card']
    
    # Check for restricted field access in the template
    for field in restricted_fields:
        if f'customer.{field}' in template_string:
            raise ValueError(f"Access to customer.{field} is restricted")
    
    # Create a context dictionary for string formatting
    context = {
        'order': order,
        'customer': customer
    }
    
    # Format the template string
    try:
        result = template_string.format(**context)
    except (KeyError, AttributeError) as e:
        raise ValueError(f"Invalid placeholder in template: {e}")
    
    return result
