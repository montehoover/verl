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
    if order_id not in purchases:
        raise ValueError(f"Invalid order ID: {order_id}")
    
    order = purchases[order_id]
    customer = order.client
    
    # Check for restricted data access attempts
    if 'credit_card' in template and '{customer.credit_card}' in template:
        raise ValueError("Attempt to access restricted customer data: credit_card")
    
    # Create a safe context for string formatting
    context = {
        'customer': {
            'name': customer.name,
            'email': customer.email,
            'address': customer.address
        },
        'order': {
            'id': order.id,
            'items': order.items,
            'total': order.total,
            'payment_method': order.payment_method
        }
    }
    
    # Format the template
    try:
        formatted_receipt = template
        
        # Replace customer placeholders
        for key, value in context['customer'].items():
            placeholder = f'{{customer.{key}}}'
            if placeholder in formatted_receipt:
                formatted_receipt = formatted_receipt.replace(placeholder, str(value))
        
        # Replace order placeholders
        for key, value in context['order'].items():
            placeholder = f'{{order.{key}}}'
            if placeholder in formatted_receipt:
                formatted_receipt = formatted_receipt.replace(placeholder, str(value))
        
        return formatted_receipt
    except Exception as e:
        raise ValueError(f"Error formatting receipt: {str(e)}")
