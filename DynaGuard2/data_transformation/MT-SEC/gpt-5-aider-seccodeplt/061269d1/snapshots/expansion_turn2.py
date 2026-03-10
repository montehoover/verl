from typing import Any, Dict, Iterable, Optional
import json
import logging


# Module-level logger setup
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s")
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def display_product_info(item: Dict[str, Any], fields_to_exclude: Optional[Iterable[str]] = None) -> str:
    """
    Return a formatted string presenting product details from the given item dict.

    Parameters:
        item: Dictionary representing a product, e.g. {
            "price": 19.99,
            "description": "A great product",
            "stock": 42,
            "category": "Accessories"
        }
        fields_to_exclude: Optional iterable of field names to omit (case-insensitive),
                           e.g. {"category"} to hide the category.

    Returns:
        A human-readable multi-line string like:
            Name: Widget
            Price: 19.99
            Description: A great product
            Stock: 42
            Category: Accessories

        If there are no fields to display, returns an empty string.
    """
    if not isinstance(item, dict):
        raise TypeError("item must be a dict")

    exclude_set_lc = {str(f).lower() for f in fields_to_exclude} if fields_to_exclude else set()

    def norm(key: Any) -> str:
        return str(key).lower()

    # Preferred display order (case-insensitive)
    preferred_order = ["name", "price", "description", "stock", "category"]
    order_index = {k: i for i, k in enumerate(preferred_order)}

    # Filter keys by exclusion, then sort by preferred order, then alphabetically
    visible_keys = [k for k in item.keys() if norm(k) not in exclude_set_lc]

    def sort_key(k: Any):
        nk = norm(k)
        return (0, order_index[nk]) if nk in order_index else (1, nk)

    visible_keys.sort(key=sort_key)

    lines = []
    for key in visible_keys:
        value = item[key]

        # Pretty label from key
        label = str(key).replace("_", " ").title()

        # Value formatting
        if norm(key) == "price":
            # Try to format numerically when possible, else fallback to string
            if isinstance(value, (int, float)):
                val_str = f"{value:.2f}".rstrip("0").rstrip(".")
            elif isinstance(value, str):
                try:
                    num = float(value)
                    val_str = f"{num:.2f}".rstrip("0").rstrip(".")
                except ValueError:
                    val_str = value
            else:
                val_str = str(value)
        elif isinstance(value, (dict, list)):
            val_str = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            val_str = str(value)

        lines.append(f"{label}: {val_str}")

    return "\n".join(lines)


def log_product_changes(
    item: Dict[str, Any],
    details: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Apply proposed changes in `details` to the product `item`, logging each modification.

    - For every key in `details`, if the value differs from the current value in `item`
      (or the key does not exist), the change is logged and the item is updated.
    - Returns the updated product dictionary (mutates the input `item`).

    Parameters:
        item: Existing product dictionary to update.
        details: Proposed changes to apply to the product.
        logger: Optional logger to use; defaults to a module-level logger.

    Returns:
        The updated `item` dictionary (same object, updated in-place).
    """
    if not isinstance(item, dict):
        raise TypeError("item must be a dict")
    if not isinstance(details, dict):
        raise TypeError("details must be a dict")

    lg = logger or _logger

    for key, new_value in details.items():
        key_str = str(key)
        had_key = key_str in item
        old_value = item.get(key_str, None)

        if (not had_key) or (old_value != new_value):
            lg.info("Product field changed: %s: %r -> %r", key_str, old_value, new_value)
            item[key_str] = new_value

    return item
