import re

# Helper class for safe formatting (retained from previous version as it's useful)
class DefaultKeyDict(dict):
    def __missing__(self, key):
        return f'{{{key}}}' # Returns the placeholder itself if key is missing

# Domain model classes
class User:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card # This field is sensitive

class Purchase:
    def __init__(self, id, client, items, total, payment_method):
        self.id = id
        self.client = client # This is a User object
        self.items = items
        self.total = total
        self.payment_method = payment_method

# Predefined purchases data
purchases = {
    "ORD001": Purchase("ORD001",
                       User("Alice Wonderland", "alice.wonder@example.com", "123 Rabbit Hole, Wonderland", "1111-2222-3333-4444"),
                       ["Tea Set", "Mad Hatter's Hat"],
                       75.20,
                       "PayPal"),
    "ORD002": Purchase("ORD002", 
                       User("John Smith", "john.smith@example.com", "456 Park Ave, Metropolis, USA", "9876-5432-1098-7654"),
                       ["Item A", "Item B"],
                       150.50,
                       "Credit Card"),
    "ORD003": Purchase("ORD003",
                       User("Carol Danvers", "carol.d@example.com", "789 Sky High, Galaxy", "0000-1111-2222-3333"),
                       ["Kree Helmet", "Flerken Toy"],
                       550.00,
                       "StarkPay")
}

def create_user_receipt(order_id, template):
    """
    Creates a formatted user receipt string for a given order ID using a template.

    Args:
        order_id (str): The ID of the purchase to retrieve.
        template (str): A string template with placeholders like {order.total} 
                        or {customer.name}.

    Returns:
        str: The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid, or if the template attempts
                    to access restricted customer data.
    """
    purchase = purchases.get(order_id)
    if not purchase:
        raise ValueError(f"Invalid order ID: {order_id}")

    restricted_keys = {
        "customer.credit_card"
        # Add other sensitive fields here, e.g., "customer.internal_notes"
    }

    placeholders_in_template = re.findall(r'\{(.+?)\}', template)
    for ph in placeholders_in_template:
        if ph in restricted_keys:
            raise ValueError(f"Attempt to access restricted field '{ph}' in template.")

    data_for_template = {
        "order.id": purchase.id,
        "order.items": ", ".join(purchase.items) if isinstance(purchase.items, list) else str(purchase.items),
        "order.total": f"{purchase.total:.2f}",
        "order.payment_method": purchase.payment_method,
        "customer.name": purchase.client.name,
        "customer.email": purchase.client.email,
        "customer.address": purchase.client.address,
    }
    
    formatter_dict = DefaultKeyDict(data_for_template)
    return template.format_map(formatter_dict)

if __name__ == '__main__':
    receipt_template_valid = (
        "--- Receipt ---\n"
        "Order ID: {order.id}\n"
        "Customer: {customer.name} <{customer.email}>\n"
        "Shipping Address: {customer.address}\n"
        "Items: {order.items}\n"
        "Total Amount: ${order.total}\n"
        "Payment Method: {order.payment_method}\n"
        "--- Thank You ---"
    )

    receipt_template_unknown_ph = (
        "Order: {order.id}\n"
        "Customer: {customer.name}\n"
        "Total: ${order.total}\n"
        "Notes: {order.internal_notes}" # {order.internal_notes} is not defined
    )

    receipt_template_restricted = (
        "Order: {order.id}\n"
        "Customer: {customer.name}\n"
        "Credit Card: {customer.credit_card}" # Attempt to access restricted field
    )

    print("Test Case 1: Valid order and template")
    try:
        receipt1 = create_user_receipt("ORD001", receipt_template_valid)
        print(receipt1)
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 30)

    print("Test Case 2: Valid order, template with unknown placeholder")
    try:
        receipt2 = create_user_receipt("ORD002", receipt_template_unknown_ph)
        print(receipt2)
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 30)

    print("Test Case 3: Valid order, template with restricted placeholder")
    try:
        receipt3 = create_user_receipt("ORD003", receipt_template_restricted)
        print(receipt3)
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 30)

    print("Test Case 4: Invalid order ID")
    try:
        receipt4 = create_user_receipt("ORD999", receipt_template_valid)
        print(receipt4)
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 30)
    
    print("Test Case 5: Another valid order")
    try:
        receipt5 = create_user_receipt("ORD003", receipt_template_valid)
        print(receipt5)
    except ValueError as e:
        print(f"Error: {e}")
