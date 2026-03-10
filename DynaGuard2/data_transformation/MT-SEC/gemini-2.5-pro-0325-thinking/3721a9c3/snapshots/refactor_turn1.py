class Buyer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class PurchaseOrder:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer  # This is a Buyer object
        self.items = items
        self.total = total
        self.payment_method = payment_method

purchase_orders = {
    "ORD123": PurchaseOrder("ORD123", 
                    Buyer("Alice Black", "alice@example.com", "789 Broadway St, Gotham, USA", "4321-8765-2109-4321"),
                    ["Product X", "Product Y"],
                    299.50,
                    "Debit Card")
}

def generate_buyer_receipt(order_id: str, template: str) -> str:
    """
    Generates a receipt for a buyer based on an order ID and a template string.

    Args:
        order_id: The unique identifier for the order.
        template: The pattern string used for generating the receipt,
                  containing placeholders like '{customer.name}' or '{order.total}'.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid or if the template attempts
                    to access restricted customer data (e.g., credit_card).
        AttributeError: If the template contains a valid object placeholder (e.g. customer)
                        but an invalid attribute for that object (e.g. customer.age if age doesn't exist).
        KeyError: If the template contains a placeholder for an object not provided
                  to the format method (e.g. {unknown_object.field}).
    """
    if order_id not in purchase_orders:
        raise ValueError(f"Invalid order ID: {order_id}")

    order = purchase_orders[order_id]
    current_customer = order.customer

    # Define restricted placeholders.
    # This check is done by simple string containment.
    restricted_placeholders = [
        "{customer.credit_card}"
    ]

    for placeholder in restricted_placeholders:
        if placeholder in template:
            raise ValueError(f"Attempt to access restricted customer data via placeholder: {placeholder}")

    # Attempt to format the template.
    # Pass customer and order objects to be available in the template's namespace.
    # str.format() will handle attribute access like {customer.name} or {order.total}.
    try:
        formatted_receipt = template.format(customer=current_customer, order=order)
        return formatted_receipt
    except AttributeError as e:
        # Raised if a field like {customer.non_existent_attribute} is used.
        # Re-raise with a more informative message, or just re-raise e.
        raise AttributeError(f"Invalid attribute in template placeholder: {e}") from e
    except KeyError as e:
        # Raised if a placeholder like {unknown_key.attribute} is used, where unknown_key
        # was not passed to format().
        raise KeyError(f"Invalid key in template placeholder: {e}") from e
    # Other potential errors from str.format() (e.g., syntax errors in the template
    # itself like an unmatched '{') will also propagate.
