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


RESTRICTED_PLACEHOLDERS = {"customer.credit_card"}


def create_order_receipt(purchase_id, receipt_template):
    """
    Create a formatted order receipt by replacing placeholders in the template with
    values from the specified order.

    Args:
        purchase_id (str): The ID of the order in the global 'orders' dict.
        receipt_template (str): The template containing placeholders.

    Returns:
        str: The formatted receipt.

    Raises:
        ValueError: If the order ID is nonexistent or a restricted placeholder is used,
                    or if placeholders are invalid/missing.
    """
    if not isinstance(purchase_id, str):
        raise ValueError("purchase_id must be a string")
    if not isinstance(receipt_template, str):
        raise ValueError("receipt_template must be a string")

    # Lookup order from the provided global 'orders' mapping
    try:
        order = orders[purchase_id]  # noqa: F821 - provided by the environment
    except KeyError:
        raise ValueError(f"Order ID '{purchase_id}' not found")

    # Block restricted placeholders
    placeholder_pattern = re.compile(r"\{([^{}]+)\}")
    for match in placeholder_pattern.finditer(receipt_template):
        token = match.group(1).strip()
        for restricted in RESTRICTED_PLACEHOLDERS:
            if token == restricted or token.startswith(restricted + "."):
                raise ValueError(f"Restricted placeholder used: '{{{token}}}'")

    # Prepare values for replacement
    values = {
        "order": {
            "id": order.id,
            "items": order.items,
            "total": order.total,
            "payment_method": order.payment_method,
        },
        "customer": {
            "name": order.customer.name,
            "email": order.customer.email,
            "address": order.customer.address,
            # 'credit_card' intentionally omitted due to restrictions
        },
    }

    return replace_placeholders(receipt_template, values)
