import re

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

def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    """
    Generates a receipt for a completed order.
    
    Args:
        purchase_id: The unique identifier of the order
        receipt_template: The template string for generating the receipt
        
    Returns:
        The formatted receipt string with placeholders replaced
        
    Raises:
        ValueError: If the order ID doesn't exist or if attempting to access restricted data
    """
    # Check if order exists
    if purchase_id not in orders:
        raise ValueError(f"Order ID {purchase_id} does not exist")
    
    order = orders[purchase_id]
    
    # Define restricted/protected fields
    restricted_fields = {'credit_card', 'creditcard', 'card'}
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, receipt_template)
    
    # Process the template
    result = receipt_template
    
    for placeholder in placeholders:
        # Split the placeholder into parts (e.g., "customer.name" -> ["customer", "name"])
        parts = placeholder.split('.')
        
        # Check for restricted data access
        for part in parts:
            if part.lower() in restricted_fields:
                raise ValueError(f"Access to restricted data '{placeholder}' is not allowed")
        
        # Get the value for the placeholder
        try:
            value = order
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    # If attribute doesn't exist, leave placeholder unchanged
                    value = f"{{{placeholder}}}"
                    break
            
            # Format the value if it's a float (for prices)
            if isinstance(value, float) and ':' in placeholder:
                # Extract format specifier
                format_match = re.search(r':(.+)$', placeholder)
                if format_match:
                    format_spec = format_match.group(1)
                    value = f"{value:{format_spec}}"
            
            # Replace the placeholder with the value
            result = result.replace(f"{{{placeholder}}}", str(value))
            
        except Exception:
            # If any error occurs during value retrieval, leave placeholder unchanged
            pass
    
    return result
