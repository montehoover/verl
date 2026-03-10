import re
import logging
from typing import Any, Dict, Optional, Set


# Configure a module-level logger that writes to the current working directory
logger = logging.getLogger("receipt_generation")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("receipt_generation.log", encoding="utf-8")
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.propagate = False


def retrieve_order(orders_map: Dict[str, Any], purchase_id: str) -> Any:
    """
    Pure function to retrieve an order by ID from a provided orders map.

    Args:
        orders_map: Mapping from order IDs to order objects.
        purchase_id: The unique identifier of the order.

    Returns:
        The order object associated with the purchase_id.

    Raises:
        ValueError: If the order ID does not exist in the provided map.
    """
    order = orders_map.get(purchase_id) if isinstance(orders_map, dict) else None
    if order is None:
        raise ValueError("Order ID does not exist")
    return order


def process_template(receipt_template: str, order: Any, restricted_customer_fields: Optional[Set[str]] = None) -> str:
    """
    Pure function that processes a receipt template by replacing placeholders
    with data obtained from the provided order object.

    Args:
        receipt_template: The template string containing placeholders.
        order: The order object providing data for replacements.
        restricted_customer_fields: Set of restricted customer fields.

    Returns:
        The formatted receipt string with placeholders replaced.

    Raises:
        ValueError: If a placeholder requests restricted customer data.
    """
    restricted_fields = restricted_customer_fields or {"credit_card"}

    def stringify(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return ", ".join(map(str, value))
        return str(value)

    # Match placeholders like {customer.name} or {order.total}
    # Avoid matching escaped braces like {{...}}
    pattern = re.compile(r'(?<!\{)\{([^{}]+)\}(?!\})')

    def resolve_placeholder(token: str) -> Optional[str]:
        token = token.strip()
        parts = token.split(".")
        if not parts:
            return None

        # {customer.*}
        if parts[0] == "customer":
            # Don't expose the raw customer object
            if len(parts) == 1:
                return None
            obj = order.customer
            for attr in parts[1:]:
                if attr in restricted_fields:
                    raise ValueError("Access to restricted customer data is not allowed")
                if not hasattr(obj, attr):
                    return None
                obj = getattr(obj, attr)
            return stringify(obj)

        # {order.*}
        if parts[0] == "order":
            # Don't expose the raw order object
            if len(parts) == 1:
                return None
            obj = order
            attrs = parts[1:]
            for i, attr in enumerate(attrs):
                # If navigating into the customer, prevent exposing it directly
                if attr == "customer":
                    # If the placeholder ends exactly at 'order.customer', leave unchanged
                    if i == len(attrs) - 1:
                        return None
                    obj = order.customer
                    continue

                # If currently inside customer, enforce restricted fields
                if obj is order.customer and attr in restricted_fields:
                    raise ValueError("Access to restricted customer data is not allowed")

                if not hasattr(obj, attr):
                    return None
                obj = getattr(obj, attr)

            return stringify(obj)

        # Unknown root; leave unchanged
        return None

    def replacer(match: re.Match) -> str:
        token = match.group(1)
        value = resolve_placeholder(token)
        return value if value is not None else match.group(0)

    return pattern.sub(replacer, receipt_template)


def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    """
    Generate a receipt by substituting placeholders in the template with
    customer and order information.

    Args:
        purchase_id: The unique identifier of the order.
        receipt_template: The template string containing placeholders.

    Returns:
        The formatted receipt string with placeholders replaced.

    Raises:
        ValueError: If the order ID does not exist, or if a placeholder
                    requests restricted customer data (e.g., credit_card).
    """
    logger.info(f"Starting receipt generation for order_id={purchase_id}")
    logger.info(f"Using template: {receipt_template}")

    # Retrieve the order dictionary from module globals
    orders_map = globals().get("orders")

    try:
        order = retrieve_order(orders_map if isinstance(orders_map, dict) else {}, purchase_id)
    except ValueError as e:
        logger.error(f"Order retrieval failed for order_id={purchase_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during order retrieval for order_id={purchase_id}: {e}")
        raise

    try:
        receipt = process_template(receipt_template, order, restricted_customer_fields={"credit_card"})
    except ValueError as e:
        logger.error(f"Template processing failed for order_id={purchase_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during template processing for order_id={purchase_id}: {e}")
        raise

    logger.info(f"Receipt generation succeeded for order_id={purchase_id}")
    return receipt
