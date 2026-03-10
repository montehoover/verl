import re
import logging
from datetime import datetime

# Configure logging
log_filename = f"receipt_generation_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def extract_placeholders(template: str) -> list:
    """Extract all placeholders from the template string."""
    placeholders = re.findall(r'\{([^}]+)\}', template)
    logger.debug(f"Extracted placeholders: {placeholders}")
    return placeholders

def validate_placeholders(placeholders: list) -> None:
    """Check if any placeholder tries to access restricted data."""
    restricted_fields = {'credit_card', 'creditcard', 'credit-card', 'cc'}
    
    for placeholder in placeholders:
        if any(restricted in placeholder.lower() for restricted in restricted_fields):
            logger.warning(f"Attempted access to restricted field: {placeholder}")
            raise ValueError(f"Access to restricted customer data is not allowed: {placeholder}")
    
    logger.debug("All placeholders validated successfully")

def build_replacement_map(order: Order) -> dict:
    """Build a dictionary of valid replacements from the order data."""
    replacements = {
        'order.id': order.id,
        'order.total': str(order.total),
        'order.payment_method': order.payment_method,
        'order.items': ', '.join(order.items),
        'customer.name': order.customer.name,
        'customer.email': order.customer.email,
        'customer.address': order.customer.address
    }
    logger.debug(f"Built replacement map with {len(replacements)} entries")
    return replacements

def get_attribute_value(order: Order, obj_name: str, attr_name: str) -> str:
    """Safely retrieve attribute value from order or customer object."""
    restricted_fields = {'credit_card', 'creditcard', 'credit-card', 'cc'}
    
    if obj_name == 'order' and hasattr(order, attr_name):
        value = str(getattr(order, attr_name))
        logger.debug(f"Retrieved order.{attr_name}: {value}")
        return value
    elif obj_name == 'customer' and hasattr(order.customer, attr_name):
        if attr_name.lower() not in restricted_fields:
            value = str(getattr(order.customer, attr_name))
            logger.debug(f"Retrieved customer.{attr_name}: {value}")
            return value
        else:
            logger.warning(f"Blocked access to restricted customer attribute: {attr_name}")
    
    return None

def process_template(template: str, placeholders: list, replacements: dict, order: Order) -> str:
    """Process the template and replace all valid placeholders."""
    result = template
    replaced_count = 0
    
    for placeholder in placeholders:
        key = placeholder.strip()
        
        if key in replacements:
            result = result.replace(f'{{{placeholder}}}', replacements[key])
            replaced_count += 1
            logger.debug(f"Replaced placeholder {{{placeholder}}} with predefined value")
        else:
            parts = key.split('.')
            if len(parts) == 2:
                obj_name, attr_name = parts
                value = get_attribute_value(order, obj_name, attr_name)
                if value is not None:
                    result = result.replace(f'{{{placeholder}}}', value)
                    replaced_count += 1
                    logger.debug(f"Replaced placeholder {{{placeholder}}} with dynamic value")
    
    logger.info(f"Processed template: replaced {replaced_count} of {len(placeholders)} placeholders")
    return result

def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    logger.info(f"Starting receipt generation for order ID: {purchase_id}")
    logger.debug(f"Template: {receipt_template[:100]}{'...' if len(receipt_template) > 100 else ''}")
    
    try:
        # Check if order exists
        if purchase_id not in orders:
            logger.error(f"Order ID {purchase_id} not found in database")
            raise ValueError(f"Order ID {purchase_id} does not exist")
        
        order = orders[purchase_id]
        logger.info(f"Found order {purchase_id} for customer: {order.customer.name}")
        
        # Extract and validate placeholders
        placeholders = extract_placeholders(receipt_template)
        validate_placeholders(placeholders)
        
        # Build replacement map
        replacements = build_replacement_map(order)
        
        # Process template
        result = process_template(receipt_template, placeholders, replacements, order)
        
        logger.info(f"Successfully generated receipt for order {purchase_id}")
        return result
        
    except ValueError as e:
        logger.error(f"ValueError during receipt generation for order {purchase_id}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during receipt generation for order {purchase_id}: {str(e)}")
        raise
