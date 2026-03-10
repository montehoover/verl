from typing import Any, Iterable, Mapping, Optional


def display_product_info(
    product_info: Mapping[str, Any],
    exclude: Optional[Iterable[str]] = None,
) -> str:
    """
    Return a formatted string displaying details of a product.

    Parameters:
        product_info: Mapping of product attributes (e.g., price, description, stock, category).
        exclude: Optional iterable of field names to exclude (case-insensitive).
                 Example: {'category'} to hide the category field.

    Returns:
        A multi-line string containing "Key: Value" pairs, or "No details available."
        if there is nothing to display.
    """
    if not isinstance(product_info, Mapping):
        raise TypeError("product_info must be a mapping/dict")

    # Normalize exclude set (case-insensitive)
    exclude_lc = {e.lower() for e in exclude} if exclude else set()

    # Map lowercase keys to original keys to facilitate case-insensitive operations
    lc_to_orig = {str(k).lower(): k for k in product_info.keys()}

    # Preferred display order for common product fields
    preferred_order = ["name", "title", "price", "description", "stock", "category"]

    keys_in_order = []

    # Add preferred keys first, if present and not excluded
    for lc in preferred_order:
        if lc in lc_to_orig and lc not in exclude_lc:
            keys_in_order.append(lc_to_orig[lc])

    # Add remaining keys (sorted alphabetically, case-insensitive), excluding already added and excluded
    already_added_lc = {k.lower() for k in keys_in_order}
    remaining_keys = [
        original_key
        for lc_key, original_key in lc_to_orig.items()
        if lc_key not in already_added_lc and lc_key not in exclude_lc
    ]
    remaining_keys.sort(key=lambda k: str(k).lower())

    keys_in_order.extend(remaining_keys)

    # Nothing to show
    if not keys_in_order:
        return "No details available."

    def humanize_key(key: Any) -> str:
        text = str(key).replace("_", " ").replace("-", " ").strip()
        # Title-case words while preserving all-caps abbreviations (e.g., SKU)
        words = text.split()
        pretty_words = [w if w.isupper() else w.capitalize() for w in words]
        return " ".join(pretty_words)

    def format_value(field_lc: str, value: Any) -> str:
        if value is None:
            return "N/A"
        if field_lc == "price" and isinstance(value, (int, float)):
            return f"{value:,.2f}"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, (list, tuple, set)):
            return ", ".join(map(str, value)) if value else "N/A"
        return str(value)

    lines = []
    for key in keys_in_order:
        lc = str(key).lower()
        val = product_info[key]
        lines.append(f"{humanize_key(key)}: {format_value(lc, val)}")

    return "\n".join(lines)
