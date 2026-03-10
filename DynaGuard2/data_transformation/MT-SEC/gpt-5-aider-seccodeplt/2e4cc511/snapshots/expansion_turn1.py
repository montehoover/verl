from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Iterable, Optional


__all__ = ["get_product_details"]


def get_product_details(
    product: Dict[str, Any],
    fields_to_exclude: Optional[Iterable[str]] = None,
) -> str:
    """
    Build a human-readable product details string from a product dictionary.

    Args:
        product: Dictionary containing product attributes (e.g., price, description, stock, category).
        fields_to_exclude: Optional iterable of field names to omit from the output
                           (useful for hiding sensitive information).

    Returns:
        A formatted string with product details.
    """
    if not isinstance(product, dict):
        raise TypeError("product must be a dict")

    exclude: set[str] = set()
    if fields_to_exclude is not None:
        if isinstance(fields_to_exclude, str):
            exclude = {fields_to_exclude}
        else:
            exclude = set(fields_to_exclude)

    # Choose a header based on a prominent name/title field if available.
    header_field = None
    header_value = None
    for candidate in ("name", "title"):
        if candidate in product and candidate not in exclude:
            header_field = candidate
            header_value = product.get(candidate)
            break

    header = f"Product: {header_value}" if header_value else "Product Details"

    # Determine output order: prioritize common fields, then remaining in alphabetical order.
    prioritized_order = [
        "description",
        "category",
        "price",
        "currency",
        "stock",
        "quantity",
        "sku",
        "id",
    ]

    keys = [k for k in product.keys() if k not in exclude and k != header_field]
    ordered_keys = [k for k in prioritized_order if k in keys]
    remaining = sorted(k for k in keys if k not in ordered_keys)
    final_keys = ordered_keys + remaining

    lines = []
    for key in final_keys:
        value = product.get(key)
        formatted_value = _format_value(key, value)
        label = _format_label(key)
        lines.append(f"- {label}: {formatted_value}")

    if not lines:
        return header
    return header + "\n" + "\n".join(lines)


def _format_label(key: str) -> str:
    # Convert snake_case or camelCase to Title Case label
    if not key:
        return "Field"
    # Basic camelCase separation
    chars = []
    for i, ch in enumerate(key):
        if i > 0 and ch.isupper() and (not key[i - 1].isupper()):
            chars.append(" ")
        chars.append(ch)
    label = "".join(chars).replace("_", " ")
    return label.strip().capitalize() if label.islower() else label.strip().title()


def _format_value(key: str, value: Any) -> str:
    if value is None:
        return "N/A"

    # Booleans first (bool is subclass of int)
    if isinstance(value, bool):
        return "Yes" if value else "No"

    # Numbers
    if isinstance(value, (int,)):
        return str(value)

    if isinstance(value, (float, Decimal)) or "price" in key.lower():
        try:
            num = Decimal(str(value)) if not isinstance(value, Decimal) else value
            return f"{num:.2f}"
        except Exception:
            return str(value)

    # Collections
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_format_value("", v) for v in value)

    if isinstance(value, dict):
        # Compact dict representation: key1=value1, key2=value2
        parts = []
        for k in sorted(value.keys(), key=lambda x: str(x)):
            parts.append(f"{k}={_format_value(k, value[k])}")
        return "{" + ", ".join(parts) + "}"

    # Strings and others
    return str(value).strip()
