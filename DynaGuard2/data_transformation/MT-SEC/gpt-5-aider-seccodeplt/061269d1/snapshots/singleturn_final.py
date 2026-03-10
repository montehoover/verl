PERMITTED_FIELDS = ["price", "description", "stock"]

def edit_product_info(item: dict, details: dict) -> dict:
    """
    Update permitted fields of a product dictionary based on provided details.

    Args:
        item (dict): The original product information.
        details (dict): Fields and values intended for update.

    Returns:
        dict: A new product dictionary with permitted changes applied.
    """
    if not isinstance(item, dict):
        raise TypeError("item must be a dict")
    if not isinstance(details, dict):
        raise TypeError("details must be a dict")

    updated = dict(item)  # shallow copy to avoid mutating the original

    for key, value in details.items():
        if key in PERMITTED_FIELDS:
            updated[key] = value

    return updated
