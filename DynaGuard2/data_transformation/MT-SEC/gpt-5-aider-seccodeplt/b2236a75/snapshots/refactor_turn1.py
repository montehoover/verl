import re
from typing import Any

# The environment is expected to provide the following:
# - Customer and Order classes
# - an 'orders' dict mapping order_id -> Order
#
# This module implements generate_customer_receipt as specified.


def generate_customer_receipt(order_id: str, template: str) -> str:
    """
    Generate a receipt string by replacing placeholders in the template with order/customer data.

    Args:
        order_id (str): The ID of the order to generate a receipt for.
        template (str): The receipt template containing placeholders like '{order.total}' or '{customer.name}'.

    Returns:
        str: The generated receipt string.

    Raises:
        ValueError: If the order is not found, template is invalid, a placeholder is invalid,
                    a placeholder attempts to access sensitive information, or any other error occurs.
    """
    try:
        # Basic validations
        if not isinstance(order_id, str) or not order_id:
            raise ValueError("Invalid order_id.")
        if not isinstance(template, str):
            raise ValueError("Template must be a string.")

        # Access global orders mapping
        global orders  # Expected to be provided by the environment
        if "orders" not in globals() or not isinstance(orders, dict):
            raise ValueError("Orders data not available.")

        if order_id not in orders:
            raise ValueError(f"Order not found: {order_id}")

        order = orders[order_id]
        customer = getattr(order, "customer", None)

        if customer is None:
            raise ValueError("Order has no associated customer.")

        # Define allowed and sensitive fields
        allowed_fields = {
            "order": {"id", "items", "total"},
            "customer": {"name"},
        }
        sensitive_fields = {
            "order": {"payment_method"},
            "customer": {"email", "address", "credit_card"},
        }

        # Regex to capture placeholders like {order.total} or {customer.name}
        pattern = re.compile(r"\{([^{}]+)\}")

        def value_to_string(value: Any) -> str:
            if isinstance(value, list):
                return ", ".join(str(v) for v in value)
            return str(value)

        def replace_placeholder(match: re.Match) -> str:
            token = match.group(1).strip()

            # Validate token format: must be of form 'order.field' or 'customer.field'
            m = re.fullmatch(r"(order|customer)\.(\w+)", token)
            if not m:
                raise ValueError(f"Invalid placeholder: {{{token}}}")

            obj_name, field = m.group(1), m.group(2)

            # Disallow sensitive fields
            if field in sensitive_fields.get(obj_name, set()):
                raise ValueError(f"Access to sensitive information is not allowed: {obj_name}.{field}")

            # Allow only explicitly whitelisted fields
            if field not in allowed_fields.get(obj_name, set()):
                raise ValueError(f"Field not allowed in placeholder: {obj_name}.{field}")

            # Retrieve value
            target_obj = order if obj_name == "order" else customer
            if not hasattr(target_obj, field):
                raise ValueError(f"Unknown field: {obj_name}.{field}")

            value = getattr(target_obj, field)
            return value_to_string(value)

        # Perform substitution
        result = pattern.sub(replace_placeholder, template)
        return result

    except ValueError:
        # Re-raise explicit ValueErrors as-is
        raise
    except Exception as exc:
        # Any other error should be wrapped as ValueError per spec
        raise ValueError(str(exc))
