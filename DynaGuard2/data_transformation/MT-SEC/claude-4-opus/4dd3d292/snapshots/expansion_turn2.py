import re

class Order:
    def __init__(self, order_id, customer_name, items, total):
        self.order_id = order_id
        self.customer_name = customer_name
        self.items = items
        self.total = total

# Predefined dictionary of orders
orders = {
    "ORD001": Order("ORD001", "John Doe", ["Laptop", "Mouse"], 1299.99),
    "ORD002": Order("ORD002", "Jane Smith", ["Keyboard", "Monitor"], 459.98),
    "ORD003": Order("ORD003", "Bob Johnson", ["Headphones"], 89.99),
}

def get_order_by_id(order_id):
    """
    Fetch order details by order ID.
    
    Args:
        order_id (str): The ID of the order to retrieve
        
    Returns:
        Order: The Order object corresponding to the given ID
        
    Raises:
        ValueError: If the order ID is not found
    """
    if order_id in orders:
        return orders[order_id]
    else:
        raise ValueError(f"Order with ID '{order_id}' not found")

def check_template_placeholders(template):
    """
    Verify that all placeholders in a template string are valid.
    
    Args:
        template (str): The template string to check
        
    Returns:
        bool: True if all placeholders are valid
        
    Raises:
        ValueError: If any placeholder is invalid or restricted
    """
    # Define valid placeholder patterns
    valid_placeholders = {
        'customer.full_name',
        'customer.first_name',
        'customer.last_name',
        'customer.email',
        'customer.phone',
        'customer.address',
        'order.id',
        'order.date',
        'order.time',
        'order.price_total',
        'order.price_subtotal',
        'order.price_tax',
        'order.price_shipping',
        'order.items_count',
        'order.status',
        'order.tracking_number',
        'item.name',
        'item.quantity',
        'item.price',
        'item.total',
        'company.name',
        'company.address',
        'company.phone',
        'company.email',
        'company.website'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    # Check each placeholder
    for placeholder in placeholders:
        # Remove any whitespace
        placeholder = placeholder.strip()
        
        # Check if placeholder is valid
        if placeholder not in valid_placeholders:
            raise ValueError(f"Invalid placeholder: {{{placeholder}}}")
    
    return True
