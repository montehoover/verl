from typing import Any, Dict
import numbers


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
