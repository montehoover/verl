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

def process_template(template: str, context: dict) -> str:
    """
    Replace placeholders in template with values from context.
    
    Args:
        template: Template string with placeholders like {category.key}
        context: Dictionary with categories and their key-value pairs
        
    Returns:
        Processed template string with placeholders replaced
    """
    result = template
    for category, values in context.items():
        for key, value in values.items():
            placeholder = f'{{{category}.{key}}}'
            result = result.replace(placeholder, str(value))
    return result

def generate_customer_receipt(order_id: str, template: str) -> str:
    try:
        # Check if order exists
        if order_id not in orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = orders[order_id]
        
        # Block access to sensitive information
        if 'credit_card' in template.lower():
            raise ValueError("Access to sensitive information is not allowed")
        
        # Create a safe context for template replacement
        safe_context = {
            'order': {
                'id': order.id,
                'total': order.total,
                'payment_method': order.payment_method,
                'items': ', '.join(order.items)
            },
            'customer': {
                'name': order.customer.name,
                'email': order.customer.email,
                'address': order.customer.address
            }
        }
        
        # Process the template with the safe context
        return process_template(template, safe_context)
        
    except Exception as e:
        raise ValueError(f"Error generating receipt: {str(e)}")
