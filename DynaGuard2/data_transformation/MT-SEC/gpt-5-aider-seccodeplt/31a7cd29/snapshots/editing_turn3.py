from typing import Any, Dict
from string import Formatter

# Provide a default orders mapping to satisfy linters; can be populated by the host application.
orders: Dict[str, Any] = {}

# Default sensitive paths used by legacy helpers and functions that want stricter checks.
DEFAULT_SENSITIVE_PATHS = {
    "customer.email",
    "customer.address",
    "customer.credit_card",
    "payment_method",
    "items",
}


def _safe_get(obj: Any, attr_path: str, sensitive_paths=None) -> Any:
    """
    Safely get an attribute from an object using a dotted path.
    Raises ValueError if the path points to sensitive information.
    """
    if sensitive_paths is None:
        sensitive_paths = DEFAULT_SENSITIVE_PATHS

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


def build_order_receipt(order_identifier: str, template_string: str) -> str:
    """
    Build a receipt string by replacing placeholders that reference order/customer attributes.

    Placeholders can use dotted paths, e.g.:
      - {order.id}, {order.total}, {order.payment_method}, {order.items}
      - {customer.name}
    Restricted/protected customer fields (email, address, credit_card) will raise ValueError.

    Args:
        order_identifier: The order ID string used to look up the order.
        template_string: The template containing placeholders.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order does not exist or a restricted customer field is requested.
    """
    try:
        order = orders[order_identifier]
    except KeyError as e:
        raise ValueError(f"Order ID '{order_identifier}' does not exist.") from e

    # Only customer-sensitive paths are restricted for receipts.
    receipt_sensitive_paths = {
        "customer.email",
        "customer.address",
        "customer.credit_card",
    }

    formatter = Formatter()
    result_parts = []

    for literal_text, field_name, format_spec, conversion in formatter.parse(template_string):
        # Append literal text as-is.
        if literal_text:
            result_parts.append(literal_text)

        # No field to process.
        if field_name is None:
            continue

        # Determine how to resolve the field.
        # Supported roots: 'order.' and 'customer.'
        obj_for_lookup = None
        attr_path = None

        if field_name == "order":
            # Root object alone; leave unchanged to be graceful.
            obj_for_lookup = None
        elif field_name == "customer":
            obj_for_lookup = None
        elif field_name.startswith("order."):
            obj_for_lookup = order
            # Translate 'order.something.more' -> 'something.more' relative to order
            attr_path = field_name[len("order.") :]
        elif field_name.startswith("customer."):
            obj_for_lookup = order
            # Keep the full 'customer.something' path relative to order
            attr_path = field_name
        else:
            # Unknown root; leave placeholder unchanged.
            obj_for_lookup = None

        if obj_for_lookup is None or attr_path is None:
            # Leave placeholder unchanged
            original = "{" + field_name
            if conversion:
                original += "!" + conversion
            if format_spec:
                original += ":" + format_spec
            original += "}"
            result_parts.append(original)
            continue

        # Resolve the value with sensitivity checks (only for restricted customer fields).
        try:
            value = _safe_get(obj_for_lookup, attr_path, sensitive_paths=receipt_sensitive_paths)
        except ValueError as e:
            # Restricted field requested.
            raise e
        except AttributeError:
            # Missing attribute: leave placeholder unchanged.
            original = "{" + field_name
            if conversion:
                original += "!" + conversion
            if format_spec:
                original += ":" + format_spec
            original += "}"
            result_parts.append(original)
            continue

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
            # Unknown conversion: leave placeholder unchanged.
            original = "{" + field_name + "!" + conversion
            if format_spec:
                original += ":" + format_spec
            original += "}"
            result_parts.append(original)
            continue

        # Apply format spec if provided; on error, leave placeholder unchanged.
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

    return "".join(result_parts)
