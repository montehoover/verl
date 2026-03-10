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

def _get_order_data(order_id: str, all_orders: dict) -> PurchaseOrder:
    """
    Retrieves order data for a given order ID.

    Args:
        order_id: The unique identifier for the order.
        all_orders: A dictionary of all purchase orders.

    Returns:
        The PurchaseOrder object.

    Raises:
        ValueError: If the order ID is invalid.
    """
    if order_id not in all_orders:
        raise ValueError(f"Invalid order ID: {order_id}")
    return all_orders[order_id]

def _format_receipt(template: str, customer: Buyer, order: PurchaseOrder) -> str:
    """
    Formats the receipt string using customer and order data.

    Args:
        template: The pattern string for the receipt.
        customer: The Buyer object.
        order: The PurchaseOrder object.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the template attempts to access restricted customer data.
        AttributeError: If the template contains an invalid attribute for an object
                        (propagated from template.format()).
        KeyError: If the template contains an invalid key
                  (propagated from template.format()).
    """
    # Define restricted placeholders.
    # This check is done by simple string containment.
    restricted_placeholders = [
        "{customer.credit_card}"
    ]

    for placeholder in restricted_placeholders:
        if placeholder in template:
            raise ValueError(f"Attempt to access restricted customer data via placeholder: {placeholder}")

    # Attempt to format the template.
    # str.format() will handle attribute access like {customer.name} or {order.total}.
    # AttributeError or KeyError from template.format() will propagate up.
    formatted_receipt = template.format(customer=customer, order=order)
    return formatted_receipt

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
    # Retrieve order data using the helper function.
    # This may raise ValueError for an invalid order_id.
    order = _get_order_data(order_id, purchase_orders)
    current_customer = order.customer

    # Attempt to format the receipt using the helper function.
    # This may raise ValueError for restricted data access,
    # or propagate AttributeError/KeyError from formatting.
    try:
        # _format_receipt handles the restricted placeholder check and formatting
        formatted_receipt = _format_receipt(template, current_customer, order)
        return formatted_receipt
    except AttributeError as e:
        # Raised if _format_receipt propagates an AttributeError
        # (e.g., from a field like {customer.non_existent_attribute}).
        # Re-raise with a more informative message.
        raise AttributeError(f"Invalid attribute in template placeholder: {e}") from e
    except KeyError as e:
        # Raised if _format_receipt propagates a KeyError
        # (e.g., from a placeholder like {unknown_key.attribute}).
        raise KeyError(f"Invalid key in template placeholder: {e}") from e
    # ValueError from _get_order_data (invalid order_id) or 
    # _format_receipt (restricted data) will propagate directly.
    # Other potential errors from str.format() within _format_receipt 
    # (e.g., syntax errors in the template itself like an unmatched '{') 
    # will also propagate.
