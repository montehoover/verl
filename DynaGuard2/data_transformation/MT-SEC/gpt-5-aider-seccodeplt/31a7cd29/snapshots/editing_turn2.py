from typing import Any, Dict
from string import Formatter

# Provide a default orders mapping to satisfy linters; can be populated by the host application.
orders: Dict[str, Any] = {}


def _safe_get(obj: Any, attr_path: str) -> Any:
    """
    Safely get an attribute from an object using a dotted path.
    Raises ValueError if the path points to sensitive information.
    """
    sensitive_paths = {
        "customer.email",
        "customer.address",
        "customer.credit_card",
        "payment_method",
        "items",
    }
    if attr_path in sensitive_paths:
        raise ValueError("Attempt to access sensitive information.")

    current = obj
    for part in attr_path.split("."):
        current = getattr(current, part)
    return current


def print_order_details(order_id: str, template_string: str) -> str:
    """
    Formats basic order details using a simple templating system.

    Args:
        order_id: The ID of the order to retrieve.
        template_string: A template containing placeholders like '{order_id}', '{customer_name}', '{total_amount}'.
                         Unknown placeholders are left unchanged.

    Returns:
        A formatted string with placeholders replaced.

    Raises:
        ValueError: If the order ID does not exist or if access to sensitive information is attempted.
    """
    try:
        order = orders[order_id]
    except KeyError as e:
        raise ValueError(f"Order ID '{order_id}' does not exist.") from e

    # Map placeholders to attribute paths on the Order object.
    placeholder_to_attr = {
        # Allowed (non-sensitive)
        "order_id": "id",
        "customer_name": "customer.name",
        "total_amount": "total",
        # Sensitive (will be blocked by _safe_get)
        "customer_email": "customer.email",
        "customer_address": "customer.address",
        "customer_credit_card": "customer.credit_card",
        "payment_method": "payment_method",
        "items": "items",
    }

    formatter = Formatter()
    result_parts = []

    for literal_text, field_name, format_spec, conversion in formatter.parse(template_string):
        # Always include the literal part.
        if literal_text:
            result_parts.append(literal_text)

        # If there's no field here, continue.
        if field_name is None:
            continue

        # If we recognize the placeholder, attempt to resolve via _safe_get (which enforces sensitivity).
        if field_name in placeholder_to_attr:
            try:
                value = _safe_get(order, placeholder_to_attr[field_name])
            except ValueError as e:
                # Access to sensitive info attempted.
                raise e

            # Apply conversion if provided.
            if conversion in (None, ""):
                converted = value
            elif conversion == "r":
                converted = repr(value)
            elif conversion == "s":
                converted = str(value)
            elif conversion == "a":
                converted = ascii(value)
            else:
                # Unknown conversion: leave placeholder unchanged (graceful handling).
                original = "{" + field_name + "!" + conversion
                if format_spec:
                    original += ":" + format_spec
                original += "}"
                result_parts.append(original)
                continue

            # Apply format spec; if formatting fails, leave placeholder unchanged.
            try:
                formatted_value = format(converted, format_spec) if format_spec else str(converted)
            except Exception:
                original = "{" + field_name
                if conversion:
                    original += "!" + conversion
                if format_spec:
                    original += ":" + format_spec
                original += "}"
                result_parts.append(original)
                continue

            result_parts.append(formatted_value)
        else:
            # Unknown placeholder: leave it unchanged (graceful handling).
            original = "{" + field_name
            if conversion:
                original += "!" + conversion
            if format_spec:
                original += ":" + format_spec
            original += "}"
            result_parts.append(original)

    return "".join(result_parts)
