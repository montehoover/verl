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

def create_receipt_for_transaction(transaction_id: str, template_format: str) -> str:
    # Check if the order exists
    if transaction_id not in orders:
        raise ValueError(f"Order ID {transaction_id} does not exist")
    
    order = orders[transaction_id]
    
    # Define restricted/protected customer fields
    restricted_fields = {'credit_card', 'creditcard', 'credit', 'card'}
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template_format)
    
    # Process the template
    result = template_format
    
    for placeholder in placeholders:
        # Split the placeholder to handle nested attributes
        parts = placeholder.split('.')
        
        # Check for restricted customer data
        if len(parts) >= 2 and parts[0] == 'customer':
            field_name = parts[1].lower().replace('_', '').replace('-', '')
            if field_name in restricted_fields:
                raise ValueError(f"Access to restricted customer data '{parts[1]}' is not allowed")
        
        # Get the value for the placeholder
        try:
            value = None
            if parts[0] == 'order':
                value = order
                for part in parts[1:]:
                    value = getattr(value, part)
            elif parts[0] == 'customer':
                value = order.customer
                for part in parts[1:]:
                    value = getattr(value, part)
            else:
                # If it's not order or customer, skip
                continue
            
            # Replace the placeholder with the actual value
            # Check if it's a formatted placeholder (e.g., {order.total:.2f})
            formatted_placeholder = '{' + placeholder + '}'
            if ':' in placeholder:
                # Extract the format specifier
                field_path, format_spec = placeholder.rsplit(':', 1)
                # Re-evaluate to get the actual value without format
                parts_without_format = field_path.split('.')
                value = None
                if parts_without_format[0] == 'order':
                    value = order
                    for part in parts_without_format[1:]:
                        value = getattr(value, part)
                elif parts_without_format[0] == 'customer':
                    value = order.customer
                    for part in parts_without_format[1:]:
                        value = getattr(value, part)
                
                # Apply the format
                formatted_value = f"{value:{format_spec}}"
                result = result.replace(formatted_placeholder, formatted_value)
            else:
                result = result.replace(formatted_placeholder, str(value))
                
        except AttributeError:
            # If attribute doesn't exist, leave placeholder as is
            continue
    
    return result
