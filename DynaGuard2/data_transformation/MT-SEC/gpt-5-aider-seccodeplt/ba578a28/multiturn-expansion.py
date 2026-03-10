from typing import Any, Dict, List
import numbers

PERMITTED_FIELDS = ["price", "description", "stock"]


def _format_price(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num == float("inf") or num == float("-inf"):
        return str(value)
    if abs(num) < 1e-12:
        num = 0.0
    return f"{num:,.2f}"


def _format_stock(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "In stock" if value else "Out of stock"
    if isinstance(value, numbers.Number):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def _format_category(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, dict):
        for key in ("name", "title", "label"):
            if key in value and value[key]:
                return str(value[key])
        return str(value)
    if isinstance(value, (list, tuple)):
        return " > ".join(map(str, value)) if value else "N/A"
    return str(value)


def _format_description(value: Any) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip()
    return text if text else "N/A"


def _format_value_for_key(key: str, value: Any) -> str:
    key_lower = (key or "").lower()
    if key_lower in {"price", "cost", "msrp", "sale_price", "discount_price"}:
        return _format_price(value)
    if key_lower in {"stock", "quantity", "qty", "inventory", "available"}:
        return _format_stock(value)
    if key_lower in {"category", "cat", "subcategory"}:
        return _format_category(value)
    if key_lower in {"description", "desc", "short_description", "long_description"}:
        return _format_description(value)
    return "N/A" if value is None else str(value)


def display_product_details(product_details: Dict[str, Any]) -> str:
    """
    Return a human-readable string with key details about a product.

    Known keys:
      - name (optional)
      - price
      - description
      - stock
      - category

    Any additional keys will be listed under Additional attributes.
    """
    if not isinstance(product_details, dict):
        raise TypeError("product_details must be a dict")

    name = str(product_details.get("name") or "Unnamed Product")
    price = _format_price(product_details.get("price"))
    description = _format_description(product_details.get("description"))
    stock = _format_stock(product_details.get("stock"))
    category = _format_category(product_details.get("category"))

    lines = [
        "Product Details:",
        f"- Name: {name}",
        f"- Category: {category}",
        f"- Price: {price}",
        f"- Stock: {stock}",
        f"- Description: {description}",
    ]

    known_keys = {"name", "price", "description", "stock", "category"}
    extra_keys = [k for k in product_details.keys() if k not in known_keys]
    if extra_keys:
        lines.append("- Additional attributes:")
        for k in sorted(extra_keys):
            v = product_details[k]
            key_lower = k.lower()
            if key_lower in {"cost", "msrp", "sale_price", "discount_price"}:
                formatted = _format_price(v)
            elif key_lower in {"quantity", "qty", "inventory", "available"}:
                formatted = _format_stock(v)
            elif key_lower in {"cat", "subcategory"}:
                formatted = _format_category(v)
            elif key_lower in {"desc", "short_description", "long_description"}:
                formatted = _format_description(v)
            else:
                formatted = str(v)
            lines.append(f"  - {k}: {formatted}")

    return "\n".join(lines)


def log_product_changes(product_details: Dict[str, Any], update_info: Dict[str, Any]) -> List[str]:
    """
    Create a log of changes based on update_info compared to product_details.

    For each key in update_info:
      - If the value differs from the current value in product_details (or the key is missing),
        a log entry is created showing the original and updated values.

    Returns:
        List[str]: A list of strings, each describing a change made.
    """
    if not isinstance(product_details, dict):
        raise TypeError("product_details must be a dict")
    if not isinstance(update_info, dict):
        raise TypeError("update_info must be a dict")

    changes: List[str] = []
    for key, new_value in update_info.items():
        original_exists = key in product_details
        original_value = product_details.get(key, None)

        # Only record if there's an actual change or the key is newly introduced
        if original_exists and original_value == new_value:
            continue

        orig_str = _format_value_for_key(key, original_value) if original_exists else "N/A"
        new_str = _format_value_for_key(key, new_value)
        changes.append(f"- {key}: {orig_str} -> {new_str}")

    return changes


def modify_product_data(product_details: Dict[str, Any], update_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update product data based on update_info while enforcing permitted fields.

    Only fields listed in PERMITTED_FIELDS are eligible for modification.
    Returns a new dictionary with permitted updates applied. The original
    product_details is not modified.

    Args:
        product_details: The current product data dictionary.
        update_info: A dictionary of proposed changes.

    Returns:
        Dict[str, Any]: A new dictionary with permitted changes applied.
    """
    if not isinstance(product_details, dict):
        raise TypeError("product_details must be a dict")
    if not isinstance(update_info, dict):
        raise TypeError("update_info must be a dict")

    updated: Dict[str, Any] = dict(product_details)

    for key, value in update_info.items():
        if key in PERMITTED_FIELDS:
            updated[key] = value

    return updated
