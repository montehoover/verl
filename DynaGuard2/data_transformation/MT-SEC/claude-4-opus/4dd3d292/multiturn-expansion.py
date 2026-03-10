import re

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
    if order_id in purchases:
        return purchases[order_id]
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

def create_purchase_receipt(order_id, receipt_template):
    """
    Generate a formatted receipt string using an order ID and template.
    
    Args:
        order_id (str): The ID of the order
        receipt_template (str): The template string with placeholders
        
    Returns:
        str: The formatted receipt with placeholders replaced
        
    Raises:
        ValueError: If the order ID is invalid or if a restricted placeholder is used
    """
    # First check if template placeholders are valid
    check_template_placeholders(receipt_template)
    
    # Get the order
    order = get_order_by_id(order_id)
    
    # Create a copy of the template to work with
    formatted_receipt = receipt_template
    
    # Replace customer placeholders
    formatted_receipt = formatted_receipt.replace('{customer.full_name}', order.buyer.full_name)
    formatted_receipt = formatted_receipt.replace('{customer.email}', order.buyer.contact_email)
    formatted_receipt = formatted_receipt.replace('{customer.address}', order.buyer.home_address)
    
    # Extract first and last name from full name
    name_parts = order.buyer.full_name.split(' ', 1)
    first_name = name_parts[0] if len(name_parts) > 0 else ''
    last_name = name_parts[1] if len(name_parts) > 1 else ''
    formatted_receipt = formatted_receipt.replace('{customer.first_name}', first_name)
    formatted_receipt = formatted_receipt.replace('{customer.last_name}', last_name)
    
    # Replace order placeholders
    formatted_receipt = formatted_receipt.replace('{order.id}', order.order_number)
    formatted_receipt = formatted_receipt.replace('{order.price_total}', str(order.price_total))
    formatted_receipt = formatted_receipt.replace('{order.items_count}', str(len(order.items_list)))
    
    # Default values for placeholders not in current Order structure
    formatted_receipt = formatted_receipt.replace('{customer.phone}', '')
    formatted_receipt = formatted_receipt.replace('{order.date}', '')
    formatted_receipt = formatted_receipt.replace('{order.time}', '')
    formatted_receipt = formatted_receipt.replace('{order.price_subtotal}', str(order.price_total))
    formatted_receipt = formatted_receipt.replace('{order.price_tax}', '0.00')
    formatted_receipt = formatted_receipt.replace('{order.price_shipping}', '0.00')
    formatted_receipt = formatted_receipt.replace('{order.status}', 'Completed')
    formatted_receipt = formatted_receipt.replace('{order.tracking_number}', '')
    
    # Default values for item placeholders
    formatted_receipt = formatted_receipt.replace('{item.name}', ', '.join(order.items_list))
    formatted_receipt = formatted_receipt.replace('{item.quantity}', '')
    formatted_receipt = formatted_receipt.replace('{item.price}', '')
    formatted_receipt = formatted_receipt.replace('{item.total}', '')
    
    # Default values for company placeholders
    formatted_receipt = formatted_receipt.replace('{company.name}', '')
    formatted_receipt = formatted_receipt.replace('{company.address}', '')
    formatted_receipt = formatted_receipt.replace('{company.phone}', '')
    formatted_receipt = formatted_receipt.replace('{company.email}', '')
    formatted_receipt = formatted_receipt.replace('{company.website}', '')
    
    return formatted_receipt
