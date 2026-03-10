from typing import Any, Dict, Iterable, List, Optional
import re
import logging

# Module-level logger with a NullHandler to avoid "No handler found" warnings in libraries
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Fields allowed to be amended by amend_product_features
PERMITTED_FIELDS = ["price", "description", "stock"]


def display_product_details(item: Dict[str, Any], fields_to_hide: Optional[Iterable[str]] = None) -> str:
    """
    Create a human-readable string of product details.

    Args:
        item: A dictionary with product fields like price, description, stock, category, etc.
        fields_to_hide: Optional iterable of field names to exclude from the output.

    Returns:
        A formatted multi-line string with the product details.
    """
    if not isinstance(item, dict):
        raise ValueError("item must be a dictionary")

    hide_set = set(fields_to_hide or [])

    def prettify_key(key: str) -> str:
        # Replace underscores and hyphens with spaces
        key = key.replace("_", " ").replace("-", " ")
        # Insert space before CamelCase capitals (e.g., "productName" -> "product Name")
        key = re.sub(r"(?<!^)(?=[A-Z])", " ", key)
        return key.strip().title()

    def format_value(field: str, value: Any) -> str:
        if value is None:
            return "N/A"
        if field == "price":
            # Format numeric price as $1,234.56
            if isinstance(value, (int, float)):
                return f"${value:,.2f}"
            # Attempt to parse numeric string
            try:
                num = float(str(value).replace(",", "").strip())
                return f"${num:,.2f}"
            except (ValueError, TypeError):
                return str(value)
        if field == "stock":
            # Prefer integer-like rendering for stock where possible
            if isinstance(value, (int, float)):
                if float(value).is_integer():
                    return f"{int(value)}"
                return f"{value}"
            return str(value)
        return str(value)

    # Determine display order: known fields first, then the rest alphabetically
    preferred_order: List[str] = ["name", "title", "description", "price", "stock", "category", "sku", "id"]
    visible_keys = [k for k in item.keys() if k not in hide_set]

    ordered_keys: List[str] = [k for k in preferred_order if k in visible_keys]
    remaining_keys = sorted([k for k in visible_keys if k not in set(ordered_keys)])

    lines: List[str] = []
    header_added = False

    for key in ordered_keys + remaining_keys:
        if not header_added:
            lines.append("Product Details")
            header_added = True
        label = prettify_key(key)
        value_str = format_value(key, item.get(key))
        lines.append(f"{label}: {value_str}")

    # If everything was hidden or item had no keys, still return a sensible header
    if not lines:
        return "Product Details"

    return "\n".join(lines)


def log_and_update_product(
    item: Dict[str, Any],
    payload: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Update a product dictionary with new values and log all modifications.

    For each key in `payload`, the product `item` will be updated (in-place) to the
    new value. If the field's value actually changes (or is newly added), a log entry
    is emitted with the before and after values.

    Args:
        item: Existing product dictionary to update.
        payload: Dictionary containing new values to apply to the product.
        logger: Optional logger to use for change logging. If not provided, a module
                logger is used.

    Returns:
        The updated product dictionary (same object as `item`).
    """
    if not isinstance(item, dict):
        raise ValueError("item must be a dictionary")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dictionary")

    log = logger or globals()["logger"]

    for field, new_value in payload.items():
        old_exists = field in item
        old_value = item.get(field, None)
        if (not old_exists) or (old_value != new_value):
            log.info("Product field updated: %s | before=%r | after=%r", field, old_value, new_value)
        item[field] = new_value

    return item


def amend_product_features(item: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Amend a product's features, allowing changes only to permitted fields.

    Args:
        item: Existing product dictionary to update.
        payload: Dictionary containing proposed updates.

    Returns:
        The updated product dictionary (same object as `item`), with only permitted
        fields modified.
    """
    if not isinstance(item, dict):
        raise ValueError("item must be a dictionary")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dictionary")

    allowed = set(PERMITTED_FIELDS)

    for field, value in payload.items():
        if field in allowed:
            item[field] = value

    return item
