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

def get_order_by_id(order_id: str, orders_dict: dict) -> Purchase:
    """Retrieve an order by its ID from the orders dictionary."""
    if order_id not in orders_dict:
        raise ValueError(f"Invalid order ID: {order_id}")
    return orders_dict[order_id]

def check_for_restricted_fields(template: str) -> None:
    """Check if the template contains restricted customer data fields."""
    if 'credit_card' in template and '{customer.credit_card}' in template:
        raise ValueError("Attempt to access restricted customer data: credit_card")

def create_formatting_context(order: Purchase) -> dict:
    """Create a safe context dictionary for template formatting."""
    return {
        'customer': {
            'name': order.client.name,
            'email': order.client.email,
            'address': order.client.address
        },
        'order': {
            'id': order.id,
            'items': order.items,
            'total': order.total,
            'payment_method': order.payment_method
        }
    }

def format_template(template: str, context: dict) -> str:
    """Format the template string with the provided context data."""
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

def create_user_receipt(order_id: str, template: str) -> str:
    # Retrieve the order
    order = get_order_by_id(order_id, purchases)
    
    # Check for restricted data access
    check_for_restricted_fields(template)
    
    # Create formatting context
    context = create_formatting_context(order)
    
    # Format and return the receipt
    try:
        return format_template(template, context)
    except Exception as e:
        raise ValueError(f"Error formatting receipt: {str(e)}")
