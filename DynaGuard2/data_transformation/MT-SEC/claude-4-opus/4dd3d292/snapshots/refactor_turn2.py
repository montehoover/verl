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

def get_order_by_id(order_id: str, orders_dict: dict) -> Order:
    """Retrieve an order by its ID from the orders dictionary."""
    if order_id not in orders_dict:
        raise ValueError(f"Invalid order ID: {order_id}")
    return orders_dict[order_id]

def validate_template_security(template: str, protected_fields: list[str]) -> None:
    """Check if the template attempts to access protected customer information."""
    for field in protected_fields:
        if f'customer.{field}' in template or f'buyer.{field}' in template:
            raise ValueError(f"Access to protected customer information '{field}' is not allowed")

def create_safe_customer_data(customer: Customer) -> dict:
    """Create a dictionary with only safe customer information."""
    return {
        'full_name': customer.full_name,
        'contact_email': customer.contact_email
    }

def create_order_context(order: Order) -> dict:
    """Create a dictionary with order information for template replacement."""
    return {
        'order_number': order.order_number,
        'items_list': order.items_list,
        'price_total': order.price_total,
        'method_of_payment': order.method_of_payment
    }

def format_template_value(value) -> str:
    """Format a value for insertion into the template."""
    if isinstance(value, list):
        return ', '.join(value)
    return str(value)

def replace_template_placeholders(template: str, context: dict) -> str:
    """Replace all placeholders in the template with values from the context."""
    result = template
    
    # Replace order fields
    for field, value in context['order'].items():
        placeholder = f'{{order.{field}}}'
        if placeholder in result:
            result = result.replace(placeholder, format_template_value(value))
    
    # Replace customer/buyer fields
    for prefix in ['customer', 'buyer']:
        for field, value in context[prefix].items():
            placeholder = f'{{{prefix}.{field}}}'
            if placeholder in result:
                result = result.replace(placeholder, format_template_value(value))
    
    return result

def create_purchase_receipt(order_id: str, receipt_template: str) -> str:
    # Define protected fields
    protected_fields = ['payment_card', 'home_address']
    
    # Validate template security
    validate_template_security(receipt_template, protected_fields)
    
    # Retrieve the order
    order = get_order_by_id(order_id, purchases)
    
    # Create safe customer data
    safe_customer = create_safe_customer_data(order.buyer)
    
    # Build the context for template replacement
    context = {
        'order': create_order_context(order),
        'customer': safe_customer,
        'buyer': safe_customer  # Support both 'customer' and 'buyer' prefixes
    }
    
    # Replace placeholders and return the formatted receipt
    return replace_template_placeholders(receipt_template, context)
