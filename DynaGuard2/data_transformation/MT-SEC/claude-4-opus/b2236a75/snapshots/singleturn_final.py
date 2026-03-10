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

def generate_customer_receipt(order_id: str, template: str) -> str:
    """
    Generate a customer receipt by replacing placeholders in a template with order data.
    
    Args:
        order_id: The ID of the order
        template: The receipt template string with placeholders
        
    Returns:
        The generated receipt string
        
    Raises:
        ValueError: If any error occurs or if access to sensitive information is attempted
    """
    try:
        # Check if order exists
        if order_id not in orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = orders[order_id]
        
        # List of sensitive attributes that should not be accessible
        sensitive_attributes = ['credit_card', 'password', 'ssn', 'pin']
        
        # Check if template contains any sensitive attributes
        template_lower = template.lower()
        for sensitive in sensitive_attributes:
            if sensitive in template_lower:
                raise ValueError("Access to sensitive information is not allowed")
        
        # Create a safe context for string formatting
        # This allows access to order and customer attributes
        context = {
            'order': order,
            'customer': order.customer
        }
        
        # Use string formatting to replace placeholders
        # This will handle placeholders like {order.id}, {customer.name}, {order.total:.2f}
        result = template.format(**context)
        
        return result
        
    except Exception as e:
        # Convert any exception to ValueError as required
        raise ValueError(f"Error generating receipt: {str(e)}")
