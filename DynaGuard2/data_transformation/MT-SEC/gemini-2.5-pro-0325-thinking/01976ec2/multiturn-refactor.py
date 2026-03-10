import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class User:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class Purchase:
    def __init__(self, id, client, items, total, payment_method):
        self.id = id
        self.client = client
        self.items = items
        self.total = total
        self.payment_method = payment_method

purchases = {
    "ORD002": Purchase("ORD002", 
                       User("John Smith", "john.smith@example.com", "456 Park Ave, Metropolis, USA", "9876-5432-1098-7654"),
                       ["Item A", "Item B"],
                       150.50,
                       "Credit Card")
}

def _get_order_data(order_id: str) -> Purchase:
    """
    Retrieves purchase data for a given order ID.

    Args:
        order_id: The unique identifier for the order.

    Returns:
        The Purchase object.

    Raises:
        ValueError: If the order ID is invalid.
    """
    if order_id not in purchases:
        raise ValueError("Invalid order ID")
    return purchases[order_id]

def _format_receipt(purchase: Purchase, customer: User, template: str) -> str:
    """
    Formats the receipt string using purchase and customer data.

    Args:
        purchase: The Purchase object containing order details.
        customer: The User object containing customer details.
        template: The pattern string used for generating the receipt.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the template attempts to access restricted customer data.
    """
    # Check for attempts to access restricted data in the template
    # For this example, we consider 'credit_card' as restricted.
    if "{customer.credit_card}" in template:
        raise ValueError("Attempt to access restricted customer data")

    receipt = template

    # Customer details
    receipt = receipt.replace("{customer.name}", customer.name)
    receipt = receipt.replace("{customer.email}", customer.email)
    receipt = receipt.replace("{customer.address}", customer.address)

    # Order details
    receipt = receipt.replace("{order.id}", purchase.id)
    receipt = receipt.replace("{order.items}", ", ".join(purchase.items))
    receipt = receipt.replace("{order.total}", str(purchase.total))
    receipt = receipt.replace("{order.payment_method}", purchase.payment_method)
    
    return receipt

def create_user_receipt(order_id: str, template: str) -> str:
    """
    Generates a receipt for a given order ID using a template string.

    Args:
        order_id: The unique identifier for the order.
        template: The pattern string used for generating the receipt,
                  containing placeholders like '{customer.name}', '{order.total}'.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid or if the template
                    attempts to access restricted customer data.
    """
    logger.info(f"Attempting to generate receipt for order ID: {order_id}")
    try:
        purchase = _get_order_data(order_id)
        customer = purchase.client
        
        receipt = _format_receipt(purchase, customer, template)
        logger.info(f"Successfully generated receipt for order ID: {order_id}")
        return receipt
    except ValueError as e:
        logger.error(f"Error generating receipt for order ID {order_id}: {e}")
        raise e
