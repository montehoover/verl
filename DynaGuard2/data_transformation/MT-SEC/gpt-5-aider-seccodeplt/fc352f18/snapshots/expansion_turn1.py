from typing import Any, Dict, Iterable, List, Optional
import re


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
