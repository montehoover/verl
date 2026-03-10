import re

ORDERS = {
    "order_001": {"customer": {"name": "Alice Smith", "email": "alice@example.com"}},
    "order_002": {"customer": {"name": "Bob Johnson", "email": "bob@example.com"}},
    "order_003": {"customer": {"name": "Charlie Davis", "email": "charlie@example.com"}},
}


def get_customer_details(order_id):
    """
    Return the customer's name and email for the given order_id.

    Args:
        order_id (str): The ID of the order.

    Returns:
        dict: A dictionary with keys 'name' and 'email'.

    Raises:
        ValueError: If the order_id is not found in the ORDERS dictionary.
    """
    try:
        customer = ORDERS[order_id]["customer"]
    except KeyError:
        raise ValueError(f"Order ID '{order_id}' not found")

    return {"name": customer.get("name"), "email": customer.get("email")}


def replace_placeholders(template, values):
    """
    Replace placeholders like '{customer.name}' in the template using the provided values dict.

    Args:
        template (str): The template string containing placeholders.
        values (dict): A dictionary of values; supports nested lookup with dot notation.

    Returns:
        str: The formatted string with all placeholders replaced.

    Raises:
        ValueError: If any placeholder is invalid or missing from the values dict.
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string")
    if not isinstance(values, dict):
        raise ValueError("Values must be a dictionary")

    pattern = re.compile(r"\{([^{}]+)\}")

    def resolve_path(path, data):
        token = path.strip()
        if not token:
            raise ValueError(f"Invalid placeholder '{{{path}}}'")

        parts = token.split(".")
        current = data
        for part in parts:
            if not part:
                raise ValueError(f"Invalid placeholder '{{{path}}}'")
            if not isinstance(current, dict):
                raise ValueError(f"Invalid placeholder '{{{path}}}'")
            if part not in current:
                raise ValueError(f"Missing value for placeholder '{{{path}}}'")
            current = current[part]
        return current

    def replacer(match):
        inner = match.group(1)
        value = resolve_path(inner, values)
        return str(value)

    return pattern.sub(replacer, template)
