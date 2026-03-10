import re

def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    """
    Generate a receipt by substituting placeholders in the template with
    customer and order information.

    Args:
        purchase_id: The unique identifier of the order.
        receipt_template: The template string containing placeholders.

    Returns:
        The formatted receipt string with placeholders replaced.

    Raises:
        ValueError: If the order ID does not exist, or if a placeholder
                    requests restricted customer data (e.g., credit_card).
    """
    # Retrieve the order dictionary from module globals
    orders_map = globals().get("orders")
    if not isinstance(orders_map, dict):
        orders_map = {}

    order = orders_map.get(purchase_id)
    if order is None:
        raise ValueError("Order ID does not exist")

    restricted_customer_fields = {"credit_card"}

    def stringify(value):
        if isinstance(value, (list, tuple)):
            return ", ".join(map(str, value))
        return str(value)

    # Match placeholders like {customer.name} or {order.total}
    # Avoid matching escaped braces like {{...}}
    pattern = re.compile(r'(?<!\{)\{([^{}]+)\}(?!\})')

    def resolve_placeholder(token: str):
        token = token.strip()
        parts = token.split(".")
        if not parts:
            return None

        # {customer.*}
        if parts[0] == "customer":
            # Don't expose the raw customer object
            if len(parts) == 1:
                return None
            obj = order.customer
            for attr in parts[1:]:
                if attr in restricted_customer_fields:
                    raise ValueError("Access to restricted customer data is not allowed")
                if not hasattr(obj, attr):
                    return None
                obj = getattr(obj, attr)
            return stringify(obj)

        # {order.*}
        if parts[0] == "order":
            # Don't expose the raw order object
            if len(parts) == 1:
                return None
            obj = order
            attrs = parts[1:]
            for i, attr in enumerate(attrs):
                # If navigating into the customer, prevent exposing it directly
                if attr == "customer":
                    # If the placeholder ends exactly at 'order.customer', leave unchanged
                    if i == len(attrs) - 1:
                        return None
                    obj = order.customer
                    continue

                # If currently inside customer, enforce restricted fields
                if obj is order.customer and attr in restricted_customer_fields:
                    raise ValueError("Access to restricted customer data is not allowed")

                if not hasattr(obj, attr):
                    return None
                obj = getattr(obj, attr)

            return stringify(obj)

        # Unknown root; leave unchanged
        return None

    def replacer(match: re.Match) -> str:
        token = match.group(1)
        value = resolve_placeholder(token)
        return value if value is not None else match.group(0)

    return pattern.sub(replacer, receipt_template)
