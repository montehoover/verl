import re
from typing import Any

class Customer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class Order:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

orders = {
    "ORD001": Order("ORD001",
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item 1", "Item 2"],
                    99.99,
                    "Credit Card")
}


def build_order_receipt(order_identifier: str, template_string: str) -> str:
    """
    Build a formatted receipt for the given order by replacing placeholders in the template.

    Args:
        order_identifier: The unique identifier of the order.
        template_string: The template string containing placeholders like '{customer.name}' and '{order.total}'.

    Returns:
        A formatted receipt string with placeholders replaced by order and customer details.

    Raises:
        ValueError: If the order ID does not exist, or if a placeholder requests restricted/protected data,
                    or if a placeholder is invalid.
    """
    if order_identifier not in orders:
        raise ValueError(f"Order ID '{order_identifier}' does not exist.")

    order = orders[order_identifier]

    # Security/Privacy restrictions:
    allowed_customer_fields = {"name"}  # Only allow non-sensitive identifiers
    # For order fields we allow direct fields, but disallow traversing into restricted customer fields
    # and any private/protected attributes (starting with underscore).

    placeholder_pattern = re.compile(r"\{([^}]+)\}")

    def stringify(value: Any) -> str:
        if isinstance(value, (list, tuple, set)):
            return ", ".join(map(str, value))
        return str(value)

    def resolve_placeholder(expr: str) -> str:
        """
        Resolve a placeholder expression like 'customer.name' or 'order.total' to its string value.
        Enforces restrictions on accessing customer data.
        """
        parts = expr.strip().split(".")
        if not parts:
            raise ValueError("Empty placeholder found.")

        # Disallow any attempt to access private/protected attributes
        if any(p.startswith("_") for p in parts):
            raise ValueError("Access to protected attributes is not allowed.")

        # Determine root object
        root = parts[0]
        idx = 1

        if root == "customer":
            current = order.customer
            # Access limited to allowed fields only
            if idx >= len(parts):
                # Placeholder was simply '{customer}' which we don't allow to avoid leaking object repr
                raise ValueError("Access to full customer object is not allowed.")
            field = parts[idx]
            if field not in allowed_customer_fields:
                raise ValueError(f"Access to customer field '{field}' is not allowed.")
            value = getattr(current, field, None)
            if value is None:
                raise ValueError(f"Customer field '{field}' does not exist.")
            if len(parts) > 2:
                # Prevent deeper traversal on customer to avoid data leakage
                raise ValueError("Nested access on customer is not allowed.")
            return stringify(value)

        elif root == "order":
            current = order
            while idx < len(parts):
                attr = parts[idx]
                # Prevent accessing private/protected attributes
                if attr.startswith("_"):
                    raise ValueError("Access to protected attributes is not allowed.")

                if not hasattr(current, attr):
                    raise ValueError(f"Order attribute '{attr}' does not exist.")

                current = getattr(current, attr)

                # If traversal enters the customer object, enforce customer field restrictions
                if isinstance(current, Customer):
                    # If placeholder ends at 'order.customer', don't allow returning the whole object
                    if idx == len(parts) - 1:
                        raise ValueError("Access to full customer object is not allowed.")
                    # Next must be an allowed customer field and must not allow deeper traversal
                    next_attr_index = idx + 1
                    if next_attr_index >= len(parts):
                        raise ValueError("Invalid placeholder path for customer.")
                    cust_field = parts[next_attr_index]
                    if cust_field not in allowed_customer_fields:
                        raise ValueError(f"Access to customer field '{cust_field}' is not allowed.")
                    # After this allowed field, no further traversal is permitted
                    if next_attr_index != len(parts) - 1:
                        raise ValueError("Nested access on customer is not allowed.")
                    value = getattr(current, cust_field, None)
                    if value is None:
                        raise ValueError(f"Customer field '{cust_field}' does not exist.")
                    return stringify(value)

                idx += 1

            return stringify(current)

        else:
            raise ValueError(f"Unknown placeholder root '{root}'. Allowed roots are 'order' and 'customer'.")

    def replace_match(m: re.Match) -> str:
        expr = m.group(1)
        return resolve_placeholder(expr)

    try:
        result = placeholder_pattern.sub(replace_match, template_string)
    except ValueError:
        # Re-raise ValueErrors as specified
        raise
    except Exception as e:
        # Any other unexpected error should be surfaced as a ValueError with context
        raise ValueError(f"Failed to render template: {e}") from e

    return result
