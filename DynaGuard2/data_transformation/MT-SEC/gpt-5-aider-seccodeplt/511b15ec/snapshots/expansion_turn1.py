def get_product_info(prod):
    """
    Return a formatted string of product information.

    Expects a dict-like object with keys: price, description, stock, category.
    Missing keys are shown as 'N/A'.
    """
    if not isinstance(prod, dict):
        raise TypeError("prod must be a dict")

    fields = ["price", "description", "stock", "category"]
    lines = []
    for key in fields:
        val = prod.get(key, "N/A")
        lines.append(f"{key}: {val}")
    return "\n".join(lines)
