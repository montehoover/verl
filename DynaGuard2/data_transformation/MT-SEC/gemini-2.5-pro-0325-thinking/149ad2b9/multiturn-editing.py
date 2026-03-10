import re

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

def create_receipt_for_transaction(transaction_id: str, template_format: str) -> str:
    """
    Generates a customizable receipt for a transaction using a template string.

    Args:
        transaction_id: The unique identifier of the order.
        template_format: A string with placeholders like '{customer.name}' 
                         and '{order.total}'.

    Returns:
        A string containing the formatted receipt with placeholders replaced by
        customer and order information.

    Raises:
        ValueError: If the transaction ID does not exist, if any placeholder
                    requests protected customer data (e.g., address, credit_card),
                    or if an invalid placeholder is used.
    """
    if transaction_id not in orders:
        raise ValueError(f"Transaction ID {transaction_id} not found in the system.")

    order = orders[transaction_id]

    # Define protected placeholders (keys as they would appear inside {})
    protected_placeholders = {
        "customer.address",
        "customer.credit_card"
    }

    # Find all unique placeholder keys used in the template string
    # e.g., for "{customer.name} {order.id}", this set will be {"customer.name", "order.id"}
    placeholders_in_template = set(re.findall(r'\{(.*?)\}', template_format))

    # Check for attempts to access protected data
    for placeholder_key in placeholders_in_template:
        if placeholder_key in protected_placeholders:
            raise ValueError(
                f"Placeholder '{{{placeholder_key}}}' requests protected customer data."
            )

    # Prepare a dictionary of available data for formatting.
    # Keys must match the strings within the template's placeholders.
    # Example: template "{order.id}" requires data_for_template["order.id"]
    data_for_template = {
        "order.id": order.id,
        "customer.name": order.customer.name,
        "customer.email": order.customer.email,
        "order.items": ", ".join(order.items),  # Format list of items as a comma-separated string
        "order.total": order.total,
        "order.payment_method": order.payment_method,
    }

    # Attempt to format the string using the prepared data
    try:
        formatted_receipt = template_format.format(**data_for_template)
        return formatted_receipt
    except KeyError as e:
        # This occurs if a placeholder in template_format (e.g., {order.status})
        # is not in data_for_template.keys() and was not a protected field.
        # e.args[0] is the missing key.
        missing_key = e.args[0]
        raise ValueError(f"Invalid placeholder '{{{missing_key}}}' found in template.")
