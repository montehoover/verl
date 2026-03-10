import logging
import re
from typing import Any, Callable, Dict, Mapping, Match

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def retrieve_order(order_id: str, purchases_dict: Mapping[str, Any]) -> Any:
    """
    Pure function: Retrieve an order object by ID from the provided purchases mapping.

    Raises:
        ValueError: If the order_id is invalid or not present in the mapping.
    """
    logger.debug("Attempting to retrieve order with ID '%s'", order_id)

    if not isinstance(order_id, str) or not order_id:
        logger.error("Invalid order ID provided: %r", order_id)
        raise ValueError("Invalid order ID")

    if not isinstance(purchases_dict, Mapping):
        logger.error("Invalid purchases mapping type: %s", type(purchases_dict).__name__)
        raise ValueError("Invalid order ID")

    try:
        order = purchases_dict[order_id]
        logger.debug("Successfully retrieved order '%s'", order_id)
        return order
    except KeyError:
        logger.error("Order not found for ID '%s'", order_id)
        raise ValueError("Invalid order ID")


def format_receipt(order: Any, receipt_template: str) -> str:
    """
    Pure function: Format the receipt by substituting placeholders using order/customer data.

    Raises:
        ValueError: If the template attempts to access protected customer information.
    """
    logger.debug(
        "Starting receipt formatting for order '%s'",
        getattr(order, "order_number", "<unknown>"),
    )

    # Regex to find {placeholder} patterns
    pattern = re.compile(r"\{([^{}]+)\}")

    # Protected info indicators - any attempt to use these should raise
    protected_indicators = ("home_address", "payment_card")

    def join_items(o: Any) -> str:
        return ", ".join(o.items_list)

    # Allowed placeholder resolution map
    allowed_placeholders: Dict[str, Callable[[Any], Any]] = {
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

    def replace(match: Match[str]) -> str:
        key = match.group(1).strip()
        logger.debug("Processing placeholder '%s'", key)

        # Block access to protected info regardless of path
        lower_key = key.lower()
        if any(indicator in lower_key for indicator in protected_indicators):
            logger.warning(
                "Attempt to access protected customer information via placeholder '%s'",
                key,
            )
            raise ValueError("Access to protected customer information is not allowed")

        resolver = allowed_placeholders.get(key)
        if resolver is None:
            logger.debug("Unknown placeholder '%s' - leaving unchanged", key)
            return match.group(0)

        try:
            value = resolver(order)
        except Exception:
            logger.exception("Error resolving placeholder '%s'", key)
            return match.group(0)

        return str(value)

    result = pattern.sub(replace, receipt_template)
    logger.debug(
        "Completed receipt formatting for order '%s'",
        getattr(order, "order_number", "<unknown>"),
    )
    return result


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
    logger.info("Generating receipt for order '%s'", order_id)

    # Obtain purchases mapping from globals, then use pure helpers for the logic
    try:
        purchases_dict = globals()["purchases"]
    except KeyError:
        logger.error("Global 'purchases' mapping not found")
        raise ValueError("Invalid order ID")

    try:
        order = retrieve_order(order_id, purchases_dict)
    except ValueError:
        logger.error("Failed to retrieve order '%s'", order_id)
        raise

    try:
        receipt = format_receipt(order, receipt_template)
    except ValueError:
        logger.error(
            "Receipt formatting failed due to protected info access for order '%s'",
            order_id,
        )
        raise

    logger.info("Receipt generation completed for order '%s'", order_id)
    return receipt
