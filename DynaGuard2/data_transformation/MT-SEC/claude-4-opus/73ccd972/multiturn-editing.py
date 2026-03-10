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

def create_order_receipt(purchase_id, receipt_template):
    """
    Creates a formatted receipt by replacing placeholders in the template with order data.
    
    Args:
        purchase_id (str): The ID of the order to create a receipt for
        receipt_template (str): Template string with placeholders like {customer.name}, {order.total}
    
    Returns:
        str: Formatted receipt string with placeholders replaced
        
    Raises:
        ValueError: If order ID doesn't exist or if attempting to access restricted customer data
    """
    if purchase_id not in orders:
        raise ValueError(f"Order ID {purchase_id} does not exist")
    
    order = orders[purchase_id]
    
    # Define restricted/protected customer attributes
    restricted_attributes = {'credit_card', 'address'}
    
    # Create a copy of the template to work with
    formatted_receipt = receipt_template
    
    # Find all placeholders in the template
    import re
    placeholders = re.findall(r'\{([^}]+)\}', receipt_template)
    
    for placeholder in placeholders:
        if placeholder.startswith('customer.'):
            attribute = placeholder.split('.', 1)[1]
            if attribute in restricted_attributes:
                raise ValueError(f"Access to customer.{attribute} is restricted")
            if hasattr(order.customer, attribute):
                value = getattr(order.customer, attribute)
                formatted_receipt = formatted_receipt.replace(f'{{{placeholder}}}', str(value))
        elif placeholder.startswith('order.'):
            attribute = placeholder.split('.', 1)[1]
            if hasattr(order, attribute):
                value = getattr(order, attribute)
                formatted_receipt = formatted_receipt.replace(f'{{{placeholder}}}', str(value))
    
    return formatted_receipt

# Example usage
if __name__ == "__main__":
    # Test successful receipt creation
    template1 = "Receipt for {customer.name}\nOrder ID: {order.id}\nTotal: ${order.total}\nPayment: {order.payment_method}"
    print(create_order_receipt("ORD001", template1))
    print()
    
    # Test with email
    template2 = "Thank you {customer.name} ({customer.email}) for your order #{order.id} totaling ${order.total}"
    print(create_order_receipt("ORD001", template2))
    print()
    
    # Test with restricted data (will raise ValueError)
    try:
        template3 = "Order for {customer.name} at {customer.address}"
        print(create_order_receipt("ORD001", template3))
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with non-existent order (will raise ValueError)
    try:
        print(create_order_receipt("ORD999", template1))
    except ValueError as e:
        print(f"Error: {e}")
