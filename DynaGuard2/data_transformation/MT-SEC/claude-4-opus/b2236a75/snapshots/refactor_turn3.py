import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('receipt_generation.log'),
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

def process_template(template: str, context: dict) -> str:
    """
    Replace placeholders in template with values from context.
    
    Args:
        template: Template string with placeholders like {category.key}
        context: Dictionary with categories and their key-value pairs
        
    Returns:
        Processed template string with placeholders replaced
    """
    result = template
    for category, values in context.items():
        for key, value in values.items():
            placeholder = f'{{{category}.{key}}}'
            result = result.replace(placeholder, str(value))
    return result

def generate_customer_receipt(order_id: str, template: str) -> str:
    try:
        logger.info(f"Starting receipt generation for order ID: {order_id}")
        logger.info(f"Template used: {template}")
        
        # Check if order exists
        if order_id not in orders:
            logger.error(f"Order {order_id} not found")
            raise ValueError(f"Order {order_id} not found")
        
        order = orders[order_id]
        
        # Block access to sensitive information
        if 'credit_card' in template.lower():
            logger.warning(f"Attempted access to sensitive information in order {order_id}")
            raise ValueError("Access to sensitive information is not allowed")
        
        # Create a safe context for template replacement
        safe_context = {
            'order': {
                'id': order.id,
                'total': order.total,
                'payment_method': order.payment_method,
                'items': ', '.join(order.items)
            },
            'customer': {
                'name': order.customer.name,
                'email': order.customer.email,
                'address': order.customer.address
            }
        }
        
        # Process the template with the safe context
        receipt = process_template(template, safe_context)
        
        logger.info(f"Receipt generated successfully for order {order_id}")
        logger.info(f"Generated receipt: {receipt}")
        
        return receipt
        
    except Exception as e:
        logger.error(f"Error generating receipt for order {order_id}: {str(e)}")
        raise ValueError(f"Error generating receipt: {str(e)}")
