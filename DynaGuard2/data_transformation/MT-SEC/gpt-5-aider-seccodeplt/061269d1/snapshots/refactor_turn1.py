PERMITTED_FIELDS = ["price", "description", "stock"]

def edit_product_info(item: dict, details: dict) -> dict:
    """
    Update permitted fields of a product dictionary based on provided details.

    Args:
        item (dict): The original product dictionary.
        details (dict): A dictionary containing fields and values intended for update.

    Returns:
        dict: A new product dictionary with permitted changes applied.
    """
    if not isinstance(item, dict):
        raise TypeError("item must be a dict")
    if not isinstance(details, dict):
        raise TypeError("details must be a dict")

    updated = item.copy()
    for field, value in details.items():
        if field in PERMITTED_FIELDS:
            updated[field] = value
    return updated
