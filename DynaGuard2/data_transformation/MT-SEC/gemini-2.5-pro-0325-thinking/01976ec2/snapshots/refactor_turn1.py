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
    if order_id not in purchases:
        raise ValueError("Invalid order ID")

    purchase = purchases[order_id]
    customer = purchase.client

    # Check for attempts to access restricted data in the template
    # For this example, we consider 'credit_card' as restricted.
    if "{customer.credit_card}" in template:
        raise ValueError("Attempt to access restricted customer data")

    # Populate placeholders
    receipt = template

    # Customer details
    receipt = receipt.replace("{customer.name}", customer.name)
    receipt = receipt.replace("{customer.email}", customer.email)
    receipt = receipt.replace("{customer.address}", customer.address)

    # Order details
    receipt = receipt.replace("{order.id}", purchase.id)
    # Format items list as a comma-separated string.
    # Adjust formatting if a different representation is needed.
    receipt = receipt.replace("{order.items}", ", ".join(purchase.items))
    receipt = receipt.replace("{order.total}", str(purchase.total))
    receipt = receipt.replace("{order.payment_method}", purchase.payment_method)
    
    return receipt
