import re

# Changes needed:
# 1) Add compose_invoice_template(invoice_id: str, receipt_format: str) to format receipts from a template.
# 2) Validate invoice_id against global `invoices` and raise ValueError if invalid.
# 3) Replace placeholders safely, allowing only a limited set:
#    - {customer.name} or {customer.full_name}
#    - {order.id}, {order.total}, {order.total_amount}, {order.items}, {order.payment_method}
#    Any other attempt to access customer fields (e.g., email, address, bank card) raises ValueError.
#
# Implementation notes:
# - We use a regex to find placeholders and substitute via a callback.
# - Unknown non-customer placeholders are left unchanged.


def compose_invoice_template(invoice_id: str, receipt_format: str) -> str:
    """
    Generate a receipt string from a template.

    Args:
        invoice_id: The identifier of the order to render.
        receipt_format: Template string containing placeholders like:
                        {customer.name}, {order.total}, {order.id}, {order.items}, {order.payment_method}

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the invoice_id is invalid or if the template attempts to access
                    protected customer information.
    """
    # Access global invoices mapping provided by the environment/setup.
    inv = globals().get("invoices")
    if not isinstance(inv, dict) or invoice_id not in inv:
        raise ValueError("Invalid order ID provided.")
    order = inv[invoice_id]

    def _replace(match: re.Match) -> str:
        token = match.group(1).strip()

        # Guard protected customer info: only allow name/full_name under customer.*
        if token.startswith("customer."):
            if token in ("customer.name", "customer.full_name"):
                return str(order.client.full_name)
            # Any other customer.* access is considered protected
            raise ValueError("Access to protected customer information is not allowed.")

        # Allowed order tokens
        if token == "order.id":
            return str(order.order_id)
        if token in ("order.total", "order.total_amount"):
            try:
                return f"{float(order.total_amount):.2f}"
            except Exception:
                return str(order.total_amount)
        if token == "order.items":
            return ", ".join(map(str, getattr(order, "item_list", [])))
        if token == "order.payment_method":
            return str(order.payment_method)

        # Leave unknown tokens unchanged
        return match.group(0)

    return re.sub(r"\{([^{}]+)\}", _replace, receipt_format)
