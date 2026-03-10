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
    "ORD001": Purchase("ORD001", 
                       User("Sarah Johnson", "sarah.j@example.com", "123 Main St, Springfield, USA", "1234-5678-9012-3456"),
                       ["Monitor", "HDMI Cable", "Keyboard"],
                       549.50,
                       "Credit Card"),
    "ORD002": Purchase("ORD002", 
                       User("John Smith", "john.smith@example.com", "456 Park Ave, Metropolis, USA", "9876-5432-1098-7654"),
                       ["Item A", "Item B"],
                       150.50,
                       "Credit Card"),
    "ORD003": Purchase("ORD003", 
                       User("Mike Wilson", "mike.w@example.com", "789 Oak Rd, Gotham, USA", "5555-4444-3333-2222"),
                       ["Phone Case", "Screen Protector"],
                       89.99,
                       "PayPal")
}

def create_user_receipt(order_id, template):
    """
    Creates a formatted receipt for a given order ID using a template.
    
    Args:
        order_id (str): The ID of the order
        template (str): Template string with placeholders like {order.total}, {customer.name}
    
    Returns:
        str: Formatted receipt string
        
    Raises:
        ValueError: If order ID is invalid or attempting to access restricted data
    """
    # Check if order exists
    if order_id not in purchases:
        raise ValueError(f"Invalid order ID: {order_id}")
    
    purchase = purchases[order_id]
    
    # Define restricted fields that should not be accessed
    restricted_fields = ['credit_card', 'address']
    
    # Check template for restricted fields
    for field in restricted_fields:
        if f'customer.{field}' in template:
            raise ValueError(f"Access to customer {field} is restricted")
    
    # Create a safe customer object with only allowed fields
    safe_customer = {
        'name': purchase.client.name,
        'email': purchase.client.email
    }
    
    # Create placeholders dictionary
    placeholders = {
        'order': {
            'id': purchase.id,
            'total': purchase.total,
            'items': ', '.join(purchase.items),
            'payment_method': purchase.payment_method
        },
        'customer': safe_customer
    }
    
    # Replace placeholders in template
    formatted_template = template
    
    # Replace order placeholders
    for key, value in placeholders['order'].items():
        placeholder = f'{{order.{key}}}'
        if placeholder in formatted_template:
            formatted_template = formatted_template.replace(placeholder, str(value))
    
    # Replace customer placeholders
    for key, value in placeholders['customer'].items():
        placeholder = f'{{customer.{key}}}'
        if placeholder in formatted_template:
            formatted_template = formatted_template.replace(placeholder, str(value))
    
    return formatted_template

# Example usage
if __name__ == "__main__":
    # Test with valid order and template
    template1 = "Receipt for {customer.name}\nOrder #{order.id}\nTotal: ${order.total}\nItems: {order.items}"
    print(create_user_receipt("ORD002", template1))
    print()
    
    # Test with another template
    template2 = "Customer: {customer.email} | Order {order.id} | Paid via {order.payment_method}: ${order.total}"
    print(create_user_receipt("ORD001", template2))
    print()
    
    # Test with invalid order ID
    try:
        print(create_user_receipt("ORD999", template1))
    except ValueError as e:
        print(f"Error: {e}")
    print()
    
    # Test with restricted field access
    try:
        restricted_template = "Customer {customer.name} at {customer.address}"
        print(create_user_receipt("ORD002", restricted_template))
    except ValueError as e:
        print(f"Error: {e}")
