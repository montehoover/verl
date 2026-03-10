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

def get_transaction_details(transaction_id: str) -> Order:
    """
    Retrieve transaction details for the given transaction ID.
    
    Args:
        transaction_id: The unique identifier of the order to retrieve.
        
    Returns:
        The Order object corresponding to the transaction ID.
        
    Raises:
        ValueError: If the transaction ID does not exist in the orders database.
    """
    if transaction_id not in orders:
        raise ValueError(f"Order ID {transaction_id} does not exist")
    return orders[transaction_id]

def validate_placeholders(placeholders: list, restricted_fields: set) -> None:
    """
    Validate that placeholders don't request restricted customer data.
    
    Args:
        placeholders: List of placeholder strings found in the template.
        restricted_fields: Set of field names that are restricted from access.
        
    Raises:
        ValueError: If any placeholder attempts to access restricted fields.
    """
    for placeholder in placeholders:
        if 'customer.' not in placeholder:
            continue
            
        field = placeholder.split('customer.')[1]
        if field in restricted_fields:
            raise ValueError(f"Access to {placeholder} is restricted")

def build_replacements(order: Order) -> dict:
    """
    Build a dictionary mapping placeholder names to their replacement values.
    
    Args:
        order: The Order object containing transaction and customer details.
        
    Returns:
        Dictionary with placeholder names as keys and their replacement values as strings.
    """
    return {
        'order.id': order.id,
        'order.total': str(order.total),
        'order.payment_method': order.payment_method,
        'order.items': ', '.join(order.items),
        'customer.name': order.customer.name,
        'customer.email': order.customer.email,
        'customer.address': order.customer.address
    }

def format_receipt(template_format: str, replacements: dict) -> str:
    """
    Format the receipt by replacing placeholders with actual values.
    
    Args:
        template_format: The template string containing placeholders in {placeholder} format.
        replacements: Dictionary mapping placeholder names to their replacement values.
        
    Returns:
        The formatted receipt string with all placeholders replaced.
    """
    result = template_format
    placeholders = re.findall(r'\{([^}]+)\}', template_format)
    
    for placeholder in placeholders:
        if placeholder not in replacements:
            continue
            
        result = result.replace(f'{{{placeholder}}}', replacements[placeholder])
    
    return result

def create_receipt_for_transaction(transaction_id: str, template_format: str) -> str:
    """
    Generate a receipt for a completed order using a template format.
    
    Args:
        transaction_id: The unique identifier of the order.
        template_format: The template string containing placeholders for order/customer data.
        
    Returns:
        The formatted receipt string with placeholders replaced by actual order details.
        
    Raises:
        ValueError: If the order ID doesn't exist or if placeholders request restricted data.
    """
    # Get transaction details
    order = get_transaction_details(transaction_id)
    
    # Define restricted/protected fields
    restricted_fields = {'credit_card'}
    
    # Parse and validate placeholders
    placeholders = re.findall(r'\{([^}]+)\}', template_format)
    validate_placeholders(placeholders, restricted_fields)
    
    # Build replacements and format receipt
    replacements = build_replacements(order)
    return format_receipt(template_format, replacements)
