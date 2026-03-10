import re
import logging

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

def extract_placeholders(template_string: str) -> list[str]:
    """Extract all placeholders from the template string."""
    return re.findall(r'\{([^}]+)\}', template_string)

def validate_placeholders(placeholders: list[str], restricted_fields: set[str]) -> None:
    """Validate that placeholders don't access restricted fields."""
    for placeholder in placeholders:
        if '.' in placeholder:
            obj_name, attr_name = placeholder.split('.', 1)
            if obj_name == 'customer' and attr_name in restricted_fields:
                raise ValueError(f"Access to restricted data '{attr_name}' is not allowed")

def build_replacements(placeholders: list[str], order: Order) -> dict[str, str]:
    """Build a dictionary of placeholder replacements from the order object."""
    replacements = {}
    
    for placeholder in placeholders:
        if '.' in placeholder:
            obj_name, attr_name = placeholder.split('.', 1)
            if obj_name == 'order':
                if hasattr(order, attr_name):
                    replacements[placeholder] = str(getattr(order, attr_name))
            elif obj_name == 'customer':
                if hasattr(order.customer, attr_name):
                    replacements[placeholder] = str(getattr(order.customer, attr_name))
        else:
            # Handle direct order attributes
            if hasattr(order, placeholder):
                replacements[placeholder] = str(getattr(order, placeholder))
    
    return replacements

def apply_replacements(template_string: str, replacements: dict[str, str]) -> str:
    """Apply the replacements to the template string."""
    result = template_string
    for placeholder, value in replacements.items():
        result = result.replace(f'{{{placeholder}}}', value)
    return result

def build_order_receipt(order_identifier: str, template_string: str) -> str:
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting receipt generation for order ID: {order_identifier}")
    logger.info(f"Template used: {template_string}")
    
    # Check if order exists
    if order_identifier not in orders:
        logger.error(f"Order ID {order_identifier} does not exist")
        raise ValueError(f"Order ID {order_identifier} does not exist")
    
    order = orders[order_identifier]
    
    # Define restricted/protected fields
    restricted_fields = {'credit_card'}
    
    # Extract all placeholders from template
    placeholders = extract_placeholders(template_string)
    logger.info(f"Extracted placeholders: {placeholders}")
    
    # Validate placeholders
    try:
        validate_placeholders(placeholders, restricted_fields)
        logger.info("Placeholder validation successful")
    except ValueError as e:
        logger.error(f"Placeholder validation failed: {str(e)}")
        raise
    
    # Build replacement dictionary
    replacements = build_replacements(placeholders, order)
    logger.info(f"Built replacements for {len(replacements)} placeholders")
    
    # Apply replacements to template
    final_receipt = apply_replacements(template_string, replacements)
    logger.info(f"Receipt generation completed successfully")
    logger.info(f"Final formatted receipt: {final_receipt}")
    
    return final_receipt
