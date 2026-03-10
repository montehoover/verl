from string import Formatter

__all__ = ["generate_buyer_receipt"]


def generate_buyer_receipt(order_id: str, template: str) -> str:
    """
    Generate a buyer receipt by formatting the provided template with order and customer data.

    Args:
        order_id: The unique identifier of the order.
        template: A template string containing placeholders like '{order.id}', '{customer.name}',
                  and standard Python format specs (e.g., '{order.total:.2f}').

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid or if the template attempts to access restricted
                    customer data (email, address, credit_card).
    """
    # Resolve the purchase_orders mapping from globals safely (avoid undefined name)
    po = globals().get("purchase_orders")
    if not isinstance(po, dict) or order_id not in po:
        raise ValueError("Invalid order ID")

    order = po[order_id]
    customer = order.customer

    # Disallow access to restricted customer attributes via the template
    restricted_customer_fields = {"email", "address", "credit_card"}
    formatter = Formatter()

    for _, field_name, _, _ in formatter.parse(template):
        if not field_name:
            continue

        # Normalize whitespace in field_name (shouldn't be present in valid format fields)
        field_name = field_name.strip()

        # Guard against attempts like 'customer[...]' (indexing) which we do not support/allow
        if field_name.startswith("customer["):
            raise ValueError("Attempt to access restricted customer data: indexing is not allowed")

        # If field is rooted at 'customer', inspect the first attribute requested
        if field_name == "customer":
            # Accessing the customer object directly is allowed
            continue

        if field_name.startswith("customer."):
            # Extract the immediate attribute requested after 'customer.'
            remainder = field_name.split(".", 1)[1]
            # Primary attribute name before any further chaining or indexing
            first_token = remainder.split(".", 1)[0].split("[", 1)[0]

            # Block direct access to restricted fields
            if first_token in restricted_customer_fields:
                raise ValueError(f"Attempt to access restricted customer data: '{first_token}'")

            # Additionally block private/dunder-like attributes
            if first_token.startswith("_"):
                raise ValueError("Attempt to access restricted customer data: private attributes are not allowed")
            if first_token.startswith("__"):
                raise ValueError("Attempt to access restricted customer data: private attributes are not allowed")

        # No special restrictions for 'order' fields per requirements

    # Perform the formatting with the allowed context
    result = template.format(order=order, customer=customer)
    return result
