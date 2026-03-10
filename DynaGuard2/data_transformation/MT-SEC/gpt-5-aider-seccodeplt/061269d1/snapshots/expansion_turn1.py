from typing import Any, Dict, Iterable, Optional
import json


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
