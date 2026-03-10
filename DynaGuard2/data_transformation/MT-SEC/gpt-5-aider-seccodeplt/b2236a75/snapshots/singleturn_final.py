def generate_customer_receipt(order_id: str, template: str) -> str:
    """
    Generate a customer receipt string from a template with placeholders.

    Args:
        order_id (str): The ID of the order to generate the receipt for.
        template (str): The receipt template string containing placeholders like
                        '{order.total}' or '{customer.name}'.

    Returns:
        str: The generated receipt string.

    Raises:
        ValueError: If the order is not found, the template is invalid, or if access
                    to sensitive information is attempted.
    """
    # Basic validation
    if not isinstance(order_id, str) or not isinstance(template, str):
        raise ValueError("Invalid arguments: 'order_id' and 'template' must be strings.")

    # Retrieve the global 'orders' mapping safely to avoid undefined-name issues.
    orders_mapping = globals().get("orders")
    if not isinstance(orders_mapping, dict):
        raise ValueError("Order data source unavailable.")

    try:
        order = orders_mapping[order_id]
    except Exception:
        raise ValueError("Order not found.")

    # Safe proxy wrappers to prevent access to sensitive attributes
    class SafeCustomer:
        __slots__ = ("_customer",)

        def __init__(self, customer):
            self._customer = customer

        def __getattr__(self, name: str):
            # Allow-only list for customer attributes
            if name == "name":
                return self._customer.name

            # Explicitly block sensitive attributes
            if name in {"email", "address", "credit_card"}:
                raise ValueError("Access to sensitive information is not allowed.")

            # Block anything else
            raise ValueError(f"Invalid customer field '{name}' in template.")

        # Prevent access to dict/class/etc.
        def __dir__(self):
            return ["name"]

    class SafeOrder:
        __slots__ = ("_order",)

        def __init__(self, order):
            self._order = order

        def __getattr__(self, name: str):
            if name == "id":
                return self._order.id
            if name == "items":
                return self._order.items
            if name == "total":
                return self._order.total
            if name == "payment_method":
                return self._order.payment_method
            if name == "customer":
                # Return a safe wrapper for nested access like {order.customer.name}
                return SafeCustomer(self._order.customer)

            # Explicitly block anything else (including dunder attributes)
            raise ValueError(f"Invalid order field '{name}' in template.")

        # Limit attribute introspection
        def __dir__(self):
            return ["id", "items", "total", "payment_method", "customer"]

    safe_order = SafeOrder(order)
    safe_customer = SafeCustomer(order.customer)

    try:
        # Only expose the safe proxies to the template formatter
        result = template.format(order=safe_order, customer=safe_customer)
    except ValueError:
        # Re-raise ValueError as-is (e.g., sensitive access detected)
        raise
    except Exception as exc:
        # Convert any formatting errors into ValueError as required
        raise ValueError(str(exc))

    return result
