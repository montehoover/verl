from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


PERMITTED_FIELDS = ["price", "description", "stock"]


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


def log_product_changes(
    product_info: Mapping[str, Any],
    changes: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Apply changes to a product dictionary and return the updated product and a log of changes.

    Parameters:
        product_info: Original product data as a mapping/dict.
        changes: Mapping of fields to new values.

    Returns:
        A tuple of:
          - updated_product: a new dict with changes applied (original is not mutated)
          - change_log: a list of { "field": str, "old": Any, "new": Any } for each changed field
    """
    if not isinstance(product_info, Mapping):
        raise TypeError("product_info must be a mapping/dict")
    if not isinstance(changes, Mapping):
        raise TypeError("changes must be a mapping/dict")

    updated: Dict[str, Any] = dict(product_info)
    change_log: List[Dict[str, Any]] = []

    for key, new_value in changes.items():
        old_value = updated.get(key, None)
        if old_value != new_value:
            updated[key] = new_value
            change_log.append({"field": str(key), "old": old_value, "new": new_value})

    return updated, change_log


def update_item_information(
    product_info: Mapping[str, Any],
    new_data: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Update product information while enforcing allowed fields.

    Only fields listed in PERMITTED_FIELDS are updated.
    Returns a new dictionary with the updates applied.
    """
    if not isinstance(product_info, Mapping):
        raise TypeError("product_info must be a mapping/dict")
    if not isinstance(new_data, Mapping):
        raise TypeError("new_data must be a mapping/dict")

    updated: Dict[str, Any] = dict(product_info)

    for field in PERMITTED_FIELDS:
        if field in new_data:
            updated[field] = new_data[field]

    return updated
