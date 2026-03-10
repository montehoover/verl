import logging
import re
from typing import Any, Mapping

# Module-level logger
logger = logging.getLogger(__name__)

# Extracted helpers for readability and maintainability.

_ALLOWED_CUSTOMER_TOKENS = {"customer.name", "customer.full_name"}


def retrieve_order(invoice_id: str, invoices_map: Mapping[str, Any]) -> Any:
    """
    Pure function that returns the order for the given invoice_id from the provided invoices map.

    Raises:
        ValueError: If invoices_map is not a mapping or the invoice_id is not present.
    """
    logger.debug("Retrieving order for invoice_id=%s", invoice_id)
    if not isinstance(invoices_map, Mapping) or invoice_id not in invoices_map:
        logger.warning("Invalid order ID access attempt: invoice_id=%s", invoice_id)
        raise ValueError("Invalid order ID provided.")
    order = invoices_map[invoice_id]
    logger.debug("Order retrieved for invoice_id=%s", invoice_id)
    return order


def format_receipt(order: Any, template: str) -> str:
    """
    Pure function that formats a receipt template using the provided order object.

    Allowed placeholders:
        - {customer.name}, {customer.full_name}
        - {order.id}, {order.total}, {order.total_amount}, {order.items}, {order.payment_method}

    Any attempt to access other customer fields raises ValueError.
    Unknown non-customer placeholders are left unchanged.
    """

    def _replace(match: re.Match) -> str:
        token = match.group(1).strip()

        # Customer fields: only allow name/full_name
        if token.startswith("customer."):
            if token in _ALLOWED_CUSTOMER_TOKENS:
                return str(order.client.full_name)
            raise ValueError("Access to protected customer information is not allowed.")

        # Order fields
        if token == "order.id":
            return str(order.order_id)
        if token in ("order.total", "order.total_amount"):
            try:
                return f"{float(order.total_amount):.2f}"
            except Exception:
                return str(order.total_amount)
        if token == "order.items":
            items = getattr(order, "item_list", [])
            return ", ".join(map(str, items))
        if token == "order.payment_method":
            return str(order.payment_method)

        # Leave unknown tokens unchanged
        return match.group(0)

    return re.sub(r"\{([^{}]+)\}", _replace, template)


def compose_invoice_template(invoice_id: str, receipt_format: str) -> str:
    """
    Orchestrates receipt generation by retrieving the order and formatting the template.

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
    logger.info("Generating receipt for invoice_id=%s", invoice_id)
    logger.debug("Using receipt template for invoice_id=%s: %s", invoice_id, receipt_format)

    inv = globals().get("invoices")
    try:
        order = retrieve_order(invoice_id, inv)
        receipt = format_receipt(order, receipt_format)
        logger.debug("Generated receipt for invoice_id=%s: %s", invoice_id, receipt)
        return receipt
    except ValueError:
        logger.exception("Failed to generate receipt for invoice_id=%s", invoice_id)
        raise
