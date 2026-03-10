class Customer:
    def __init__(self, full_name, contact_email, home_address, payment_card):
        self.full_name = full_name
        self.contact_email = contact_email
        self.home_address = home_address
        self.payment_card = payment_card

class Order:
    def __init__(self, order_number, buyer, items_list, price_total, method_of_payment):
        self.order_number = order_number
        self.buyer = buyer
        self.items_list = items_list
        self.price_total = price_total
        self.method_of_payment = method_of_payment

purchases = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item A", "Item B"],
                    99.99,
                    "Credit Card")
}

def create_purchase_receipt(order_id: str, receipt_template: str) -> str:
    # Check if order exists
    if order_id not in purchases:
        raise ValueError(f"Invalid order ID: {order_id}")
    
    order = purchases[order_id]
    customer = order.buyer
    
    # Protected attributes that should not be accessible
    protected_attributes = {'payment_card'}
    
    # Check for protected customer information in the template
    import re
    customer_placeholders = re.findall(r'\{customer\.(\w+)\}', receipt_template)
    for attr in customer_placeholders:
        if attr in protected_attributes:
            raise ValueError(f"Access to protected customer information '{attr}' is not allowed")
    
    # Create a dictionary with all available replacements
    replacements = {}
    
    # Add order attributes
    for attr in ['order_number', 'price_total', 'method_of_payment']:
        if hasattr(order, attr):
            replacements[f'order.{attr}'] = getattr(order, attr)
    
    # Add items_list as a formatted string
    if hasattr(order, 'items_list'):
        replacements['order.items_list'] = ', '.join(order.items_list)
    
    # Add customer attributes (excluding protected ones)
    for attr in ['full_name', 'contact_email', 'home_address']:
        if hasattr(customer, attr):
            replacements[f'customer.{attr}'] = getattr(customer, attr)
    
    # Replace placeholders in the template
    result = receipt_template
    for key, value in replacements.items():
        # Handle both regular placeholders and formatted placeholders (e.g., {order.price_total:.2f})
        # First, try to replace formatted placeholders
        pattern = r'\{' + re.escape(key) + r':[^}]+\}'
        matches = re.findall(pattern, result)
        for match in matches:
            # Extract the format specifier
            format_spec = match[match.index(':') + 1:-1]
            try:
                formatted_value = f"{value:{format_spec}}"
                result = result.replace(match, formatted_value)
            except:
                # If formatting fails, just use the string representation
                result = result.replace(match, str(value))
        
        # Then replace regular placeholders
        result = result.replace(f'{{{key}}}', str(value))
    
    return result
