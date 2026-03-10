import re

class Order:
    def __init__(self, transaction_id, customer_name, items, total):
        self.transaction_id = transaction_id
        self.customer_name = customer_name
        self.items = items
        self.total = total

# Example dictionary of orders (would be populated with actual data)
orders = {
    "TXN001": Order("TXN001", "John Doe", ["Item1", "Item2"], 99.99),
    "TXN002": Order("TXN002", "Jane Smith", ["Item3"], 49.99),
    "TXN003": Order("TXN003", "Bob Johnson", ["Item4", "Item5", "Item6"], 199.99)
}

def get_order_by_id(transaction_id):
    """
    Retrieves an Order object by transaction ID.
    
    Args:
        transaction_id: The transaction ID to look up
        
    Returns:
        Order object corresponding to the transaction ID
        
    Raises:
        ValueError: If transaction_id is None, empty, or not found in orders
    """
    if not transaction_id:
        raise ValueError("Transaction ID cannot be None or empty")
    
    if transaction_id not in orders:
        raise ValueError(f"Transaction ID '{transaction_id}' not found")
    
    return orders[transaction_id]

def check_template_placeholders(template):
    """
    Validates that all placeholders in a template string are permitted.
    
    Args:
        template: Template string containing placeholders like {customer.name}
        
    Raises:
        ValueError: If template contains invalid or sensitive placeholders
    """
    # Define allowed placeholders
    allowed_placeholders = {
        'customer.name',
        'customer.first_name',
        'customer.last_name',
        'order.total',
        'order.id',
        'order.transaction_id',
        'order.date',
        'order.items',
        'order.item_count',
        'shop.name',
        'shop.address',
        'shop.phone',
        'shop.email'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    found_placeholders = re.findall(placeholder_pattern, template)
    
    # Check each placeholder
    for placeholder in found_placeholders:
        # Strip whitespace
        placeholder = placeholder.strip()
        
        # Check if placeholder is in allowed list
        if placeholder not in allowed_placeholders:
            # Check for sensitive data patterns
            sensitive_patterns = [
                'password', 'pwd', 'secret', 'token', 'api_key', 'apikey',
                'credit_card', 'creditcard', 'cc_number', 'cvv', 'ssn',
                'social_security', 'bank_account', 'routing_number'
            ]
            
            placeholder_lower = placeholder.lower()
            if any(pattern in placeholder_lower for pattern in sensitive_patterns):
                raise ValueError(f"Sensitive data placeholder not allowed: {{{placeholder}}}")
            else:
                raise ValueError(f"Invalid placeholder: {{{placeholder}}}")
