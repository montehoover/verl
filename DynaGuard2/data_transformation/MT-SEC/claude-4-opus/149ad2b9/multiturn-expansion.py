import re
from datetime import datetime

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
        self.transaction_id = id  # Adding for compatibility
        self.date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Example dictionary of orders (would be populated with actual data)
orders = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item 1", "Item 2"],
                    99.99,
                    "Credit Card"),
    "TXN001": Order("TXN001", 
                    Customer("John Doe", "john@example.com", "456 Oak Ave, Somewhere, USA", "9876-5432-1098-7654"),
                    ["Item1", "Item2"], 
                    99.99,
                    "Credit Card"),
    "TXN002": Order("TXN002", 
                    Customer("Jane Smith", "jsmith@example.com", "789 Pine Rd, Elsewhere, USA", "1111-2222-3333-4444"),
                    ["Item3"], 
                    49.99,
                    "Debit Card"),
    "TXN003": Order("TXN003", 
                    Customer("Bob Johnson", "bob@example.com", "321 Elm St, Nowhere, USA", "5555-6666-7777-8888"),
                    ["Item4", "Item5", "Item6"], 
                    199.99,
                    "PayPal")
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
        'customer.email',
        'order.total',
        'order.id',
        'order.transaction_id',
        'order.date',
        'order.items',
        'order.item_count',
        'order.payment_method',
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
                'social_security', 'bank_account', 'routing_number', 'address'
            ]
            
            placeholder_lower = placeholder.lower()
            if any(pattern in placeholder_lower for pattern in sensitive_patterns):
                raise ValueError(f"Sensitive data placeholder not allowed: {{{placeholder}}}")
            else:
                raise ValueError(f"Invalid placeholder: {{{placeholder}}}")

def create_receipt_for_transaction(transaction_id, template_format):
    """
    Creates a receipt string by substituting placeholders with actual order and customer details.
    
    Args:
        transaction_id: The transaction ID to create a receipt for
        template_format: Template string with placeholders
        
    Returns:
        Completed receipt string with placeholders replaced
        
    Raises:
        ValueError: If transaction ID doesn't exist or template contains prohibited placeholders
    """
    # Validate template placeholders first
    check_template_placeholders(template_format)
    
    # Get the order
    order = get_order_by_id(transaction_id)
    
    # Prepare replacement values
    replacements = {
        '{customer.name}': order.customer.name,
        '{customer.first_name}': order.customer.name.split()[0] if order.customer.name else '',
        '{customer.last_name}': order.customer.name.split()[-1] if order.customer.name and len(order.customer.name.split()) > 1 else '',
        '{customer.email}': order.customer.email,
        '{order.total}': f"{order.total:.2f}",
        '{order.id}': order.id,
        '{order.transaction_id}': order.id,
        '{order.date}': order.date,
        '{order.items}': ', '.join(order.items),
        '{order.item_count}': str(len(order.items)),
        '{order.payment_method}': order.payment_method,
        '{shop.name}': 'Our Shop',
        '{shop.address}': '100 Commerce St, Business City, USA',
        '{shop.phone}': '555-0123',
        '{shop.email}': 'info@ourshop.com'
    }
    
    # Replace placeholders in template
    result = template_format
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    
    return result
