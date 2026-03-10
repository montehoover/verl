import string
from typing import Any


class SafeReceiptFormatter(string.Formatter):
    """
    A safe formatter that:
    - Allows standard Python format strings with attribute access (e.g., {order.total:.2f}).
    - Blocks access to restricted customer fields.
    """

    restricted_customer_fields = {"email", "address", "credit_card"}

    def get_field(self, field_name: str, args: Any, kwargs: Any):
        """
        Intercept field resolution to restrict access to protected customer fields.
        This still delegates resolution and formatting to the base implementation.
        """
        # If the field path starts with 'customer', inspect attribute path
        # and block restricted attributes.
        if field_name.startswith("customer"):
            # Consider attribute path via dots (e.g., customer.email, customer.name)
            # Note: Indexing (e.g., customer['email']) is not expected in inputs,
            # but if present, the default resolution would error out anyway.
            parts = field_name.split(".")
            # parts[0] == 'customer'; check subsequent attributes
            for attr in parts[1:]:
                # If formatting like {customer.name!r} or {customer.name:s} occurs,
                # attr remains 'name' here; conversion/spec handled separately.
                if attr in self.restricted_customer_fields:
                    raise ValueError(f"Access to protected customer field '{attr}' is not allowed")

        try:
            return super().get_field(field_name, args, kwargs)
        except (AttributeError, KeyError, IndexError) as e:
            # Normalize to ValueError for clearer API surface
            raise ValueError(f"Invalid placeholder or path: '{field_name}'") from e


def build_order_receipt(order_identifier: str, template_string: str) -> str:
    """
    Build a receipt string by replacing placeholders in the provided template with
    order and customer details.

    Args:
        order_identifier: Unique identifier of the order to render.
        template_string: Template containing placeholders such as:
            - {order.id}, {order.total:.2f}, {order.payment_method}
            - {customer.name} (Only 'name' is allowed for customer; 'email', 'address',
              and 'credit_card' are restricted.)

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID does not exist, or if the template attempts to
        access restricted/protected customer data, or references invalid placeholders.
    """
    # Expecting a globally available 'orders' dictionary and Order/Customer classes
    try:
        orders_dict = globals()["orders"]
    except KeyError as e:
        raise ValueError("Orders data is not available in the current context") from e

    order = orders_dict.get(order_identifier)
    if order is None:
        raise ValueError(f"Order ID '{order_identifier}' does not exist")

    customer = getattr(order, "customer", None)
    if customer is None:
        # Defensive: Orders should always have a customer
        raise ValueError(f"Order '{order_identifier}' has no associated customer")

    formatter = SafeReceiptFormatter()
    try:
        # Provide only the approved roots
        result = formatter.format(template_string, order=order, customer=customer)
    except ValueError:
        # Re-raise ValueError as-is to satisfy the API contract
        raise
    except Exception as e:
        # Any other formatting error should be expressed as ValueError to the caller
        raise ValueError("Failed to render the receipt due to an unexpected formatting error") from e

    return result
