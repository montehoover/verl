import re
import logging
from typing import Any

# The environment is expected to provide the following:
# - Customer and Order classes
# - an 'orders' dict mapping order_id -> Order
#
# This module implements generate_customer_receipt as specified.

# Configure module-level logger writing to a file in the current working directory
logger = logging.getLogger("receipt_generation")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("receipt_generation.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

# Module-level constants for configuration and reuse
ALLOWED_FIELDS = {
    "order": {"id", "items", "total"},
    "customer": {"name"},
}
SENSITIVE_FIELDS = {
    "order": {"payment_method"},
    "customer": {"email", "address", "credit_card"},
}
PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")


def _value_to_string(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _render_receipt_template(template: str, order: Any, customer: Any) -> str:
    """
    Pure function that renders the given template using provided order and customer data.
    It does not access globals or mutate inputs. Raises ValueError for any invalid/sensitive placeholder.
    """
    def replace_placeholder(match: re.Match) -> str:
        token = match.group(1).strip()

        # Validate token format: must be of form 'order.field' or 'customer.field'
        m = re.fullmatch(r"(order|customer)\.(\w+)", token)
        if not m:
            raise ValueError(f"Invalid placeholder: {{{token}}}")

        obj_name, field = m.group(1), m.group(2)

        # Disallow sensitive fields
        if field in SENSITIVE_FIELDS.get(obj_name, set()):
            raise ValueError(f"Access to sensitive information is not allowed: {obj_name}.{field}")

        # Allow only explicitly whitelisted fields
        if field not in ALLOWED_FIELDS.get(obj_name, set()):
            raise ValueError(f"Field not allowed in placeholder: {obj_name}.{field}")

        # Retrieve value
        target_obj = order if obj_name == "order" else customer
        if not hasattr(target_obj, field):
            raise ValueError(f"Unknown field: {obj_name}.{field}")

        value = getattr(target_obj, field)
        return _value_to_string(value)

    return PLACEHOLDER_PATTERN.sub(replace_placeholder, template)


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
        # Log the attempt upfront for traceability
        logger.info("Starting receipt generation: order_id=%s, template=%r", order_id, template)

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

        # Delegate template rendering to pure function
        result = _render_receipt_template(template, order, customer)

        # Log the final generated receipt
        logger.info("Receipt generated: order_id=%s, receipt=%r", order_id, result)

        return result

    except ValueError as e:
        # Log validation or business-rule failures without exposing sensitive data
        logger.warning("Receipt generation failed: order_id=%s, error=%s", order_id, str(e))
        raise
    except Exception as exc:
        # Log unexpected errors with traceback, then wrap as ValueError per spec
        logger.exception("Unexpected error during receipt generation: order_id=%s", order_id)
        raise ValueError(str(exc))
