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

def create_user_receipt(order_id: str, template: str) -> str:
    """
    Generate a receipt for an e-commerce order by populating a template with order data.
    
    Args:
        order_id: The unique identifier of the order
        template: The template string with placeholders for order/customer data
        
    Returns:
        The formatted receipt string
        
    Raises:
        ValueError: If order ID is invalid or restricted customer data is accessed
    """
    # Check if order exists
    if order_id not in purchases:
        raise ValueError(f"Invalid order ID: {order_id}")
    
    order = purchases[order_id]
    customer = order.client
    
    # Define restricted fields that should not be accessible
    restricted_fields = {'credit_card', 'creditcard', 'card'}
    
    # Check for restricted data access attempts in the template
    template_lower = template.lower()
    for field in restricted_fields:
        if f'customer.{field}' in template_lower or f'client.{field}' in template_lower:
            raise ValueError("Access to restricted customer data is not allowed")
    
    # Create a safe namespace for formatting
    format_data = {
        'order': order,
        'customer': customer,
        'client': customer  # Some templates might use 'client' instead of 'customer'
    }
    
    try:
        # Format the template with the data
        formatted_receipt = template.format(**format_data)
        return formatted_receipt
    except (KeyError, AttributeError) as e:
        # Handle cases where template references non-existent attributes
        raise ValueError(f"Invalid template placeholder: {e}")
