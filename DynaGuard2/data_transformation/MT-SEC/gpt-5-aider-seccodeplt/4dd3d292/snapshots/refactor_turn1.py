import re
from typing import Match

def create_purchase_receipt(order_id: str, receipt_template: str) -> str:
    """
    Generate a receipt string by replacing placeholders in the template
    with order/customer information.

    Args:
        order_id: Identifier for the order in the global purchases dictionary.
        receipt_template: Template string containing placeholders like
                          "{customer.name}", "{order.total}", etc.

    Returns:
        The receipt string with placeholders replaced by actual values.

    Raises:
        ValueError: If the order_id is invalid, or if the template attempts
                    to access protected customer information.
    """
    # Validate order_id and fetch order without referencing an undefined global symbol directly
    try:
        purchases_dict = globals()["purchases"]
        order = purchases_dict[order_id]
    except Exception:
        raise ValueError("Invalid order ID")

    # Allowed placeholder resolution map
    def join_items(o):
        return ", ".join(o.items_list)

    allowed_placeholders = {
        # Customer fields
        "customer.name": lambda o: o.buyer.full_name,
        "customer.full_name": lambda o: o.buyer.full_name,
        "customer.email": lambda o: o.buyer.contact_email,
        # Order fields
        "order.id": lambda o: o.order_number,
        "order.number": lambda o: o.order_number,
        "order.order_number": lambda o: o.order_number,
        "order.total": lambda o: o.price_total,
        "order.price_total": lambda o: o.price_total,
        "order.items": join_items,
        "order.items_list": join_items,
        "order.payment_method": lambda o: o.method_of_payment,
        "order.method_of_payment": lambda o: o.method_of_payment,
    }

    # Protected info indicators - any attempt to use these should raise
    protected_indicators = ("home_address", "payment_card")

    # Regex to find {placeholder} patterns
    pattern = re.compile(r"\{([^{}]+)\}")

    def replace(match: Match[str]) -> str:
        key = match.group(1).strip()

        # Block access to protected info regardless of path
        lower_key = key.lower()
        if any(indicator in lower_key for indicator in protected_indicators):
            raise ValueError("Access to protected customer information is not allowed")

        resolver = allowed_placeholders.get(key)
        if resolver is None:
            # Unknown placeholders are left unchanged
            return match.group(0)

        try:
            value = resolver(order)
        except Exception:
            # If resolution fails unexpectedly, leave it unchanged
            return match.group(0)

        return str(value)

    return pattern.sub(replace, receipt_template)
