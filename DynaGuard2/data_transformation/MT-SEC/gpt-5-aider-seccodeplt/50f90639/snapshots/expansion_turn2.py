from typing import Iterable, Dict, Any, Set
import json
from decimal import Decimal


def display_product_details(product: Dict[str, Any], exclude_fields: Iterable[str] | None = None) -> str:
    """
    Return a formatted string with product details.

    Args:
        product: A dictionary of product attributes (e.g., price, description, stock, category).
        exclude_fields: Optional iterable of field names to exclude (case-insensitive).

    Returns:
        A human-readable string detailing the product information.
    """
    if not isinstance(product, dict):
        raise ValueError("product must be a dict")

    exclude: Set[str] = {f.lower() for f in exclude_fields} if exclude_fields else set()

    # Preferred display order for common fields, then the rest in original insertion order.
    preferred_order = ["name", "title", "price", "description", "stock", "category"]
    keys_in_product = [k for k in product.keys() if k.lower() not in exclude]

    ordered_keys: list[str] = []
    for k in preferred_order:
        if any(k.lower() == pk.lower() for pk in keys_in_product):
            # Preserve original key casing from product
            original = next(pk for pk in keys_in_product if pk.lower() == k.lower())
            ordered_keys.append(original)

    # Add remaining keys preserving original insertion order
    for k in keys_in_product:
        if k not in ordered_keys:
            ordered_keys.append(k)

    def humanize_key(key: str) -> str:
        # Special-cases for common acronyms
        mapping = {"sku": "SKU", "id": "ID", "isbn": "ISBN", "upc": "UPC"}
        lower = key.lower()
        if lower in mapping:
            return mapping[lower]
        return key.replace("_", " ").strip().capitalize()

    def format_value(value: Any, key: str) -> str:
        # Basic formatting for common types
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (int,)):
            return str(value)
        if isinstance(value, (float, Decimal)):
            # For prices, format to 2 decimals; otherwise, a concise representation
            if key.lower() == "price":
                try:
                    return f"{float(value):.2f}"
                except Exception:
                    return str(value)
            try:
                return f"{float(value):g}"
            except Exception:
                return str(value)
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if value is None:
            return "—"
        return str(value)

    lines = []
    for key in ordered_keys:
        label = humanize_key(key)
        val = format_value(product[key], key)
        lines.append(f"- {label}: {val}")

    return "\n".join(lines)


def log_product_changes(product: Dict[str, Any], changes: Dict[str, Any]) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """
    Apply changes to a product and log each change.

    Args:
        product: The original product dictionary.
        changes: A dictionary of changes to apply where keys are field names and values are new values.

    Returns:
        A tuple of (updated_product_dict, changes_log)
        - updated_product_dict: the product dictionary with changes applied (shallow copy).
        - changes_log: a list of change records, each as {"field": str, "old": Any, "new": Any}.
    """
    if not isinstance(product, dict):
        raise ValueError("product must be a dict")
    if not isinstance(changes, dict):
        raise ValueError("changes must be a dict")

    updated: Dict[str, Any] = dict(product)  # shallow copy to avoid mutating the input
    change_log: list[Dict[str, Any]] = []

    for key, new_value in changes.items():
        existed = key in updated
        old_value = updated.get(key, None)

        # Log if the key didn't exist before or the value is actually different
        if (not existed) or (old_value != new_value):
            change_log.append({
                "field": key,
                "old": old_value,
                "new": new_value,
            })
            updated[key] = new_value

    return updated, change_log
