# E-commerce order details retrieval system

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

# Predefined orders
orders = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item 1", "Item 2"],
                    99.99,
                    "Credit Card"),
    "ORD002": Order("ORD002",
                    Customer("John Smith", "john@example.com", "456 Oak Ave, Somewhere, USA", "9876-5432-1098-7654"),
                    ["Item 3", "Item 4", "Item 5"],
                    150.50,
                    "PayPal"),
    "ORD003": Order("ORD003",
                    Customer("Bob Johnson", "bob@example.com", "789 Pine Rd, Elsewhere, USA", "5555-4444-3333-2222"),
                    ["Item 6"],
                    25.00,
                    "Debit Card")
}

def generate_customer_receipt(order_id, template):
    """
    Generates a customer receipt by replacing placeholders in a template with order data.
    
    Args:
        order_id (str): The unique identifier for the order
        template (str): Template string with placeholders like {order.total} or {customer.name}
    
    Returns:
        str: Generated receipt string
        
    Raises:
        ValueError: If order not found, invalid placeholder, or sensitive information access attempted
    """
    # Check if order exists
    if order_id not in orders:
        raise ValueError(f"Order {order_id} not found")
    
    order = orders[order_id]
    
    # List of sensitive fields that should not be accessible
    sensitive_fields = ['credit_card', 'address']
    
    # Check template for sensitive information access
    for field in sensitive_fields:
        if f'customer.{field}' in template:
            raise ValueError(f"Access to sensitive information '{field}' is not allowed")
    
    # Create replacement dictionary
    replacements = {
        'order.id': order.id,
        'order.total': str(order.total),
        'order.payment_method': order.payment_method,
        'order.items': ', '.join(order.items),
        'customer.name': order.customer.name,
        'customer.email': order.customer.email
    }
    
    # Replace placeholders in template
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(f'{{{placeholder}}}', value)
    
    # Check if there are any unreplaced placeholders
    import re
    remaining_placeholders = re.findall(r'\{[^}]+\}', result)
    if remaining_placeholders:
        invalid_placeholders = [p for p in remaining_placeholders if p.strip('{}') not in replacements]
        if invalid_placeholders:
            raise ValueError(f"Invalid placeholder(s): {', '.join(invalid_placeholders)}")
    
    return result

# Example usage
if __name__ == "__main__":
    # Test with a basic receipt template
    template1 = """
    Receipt for Order #{order.id}
    Customer: {customer.name}
    Email: {customer.email}
    Items: {order.items}
    Total: ${order.total}
    Payment Method: {order.payment_method}
    """
    
    try:
        receipt = generate_customer_receipt("ORD001", template1)
        print(receipt)
        print("-" * 50)
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with a simple template
    template2 = "Thank you {customer.name}! Your order #{order.id} totaling ${order.total} has been processed."
    
    try:
        receipt = generate_customer_receipt("ORD002", template2)
        print(receipt)
        print("-" * 50)
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with sensitive information (should raise error)
    template3 = "Order for {customer.name} at {customer.address}"
    
    try:
        receipt = generate_customer_receipt("ORD001", template3)
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")
        print("-" * 50)
    
    # Test with non-existent order
    try:
        receipt = generate_customer_receipt("ORD999", template1)
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")
