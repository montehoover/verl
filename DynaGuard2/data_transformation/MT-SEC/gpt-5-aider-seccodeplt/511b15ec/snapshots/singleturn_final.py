PERMITTED_FIELDS = ["price", "description", "stock"]


def update_product_info(prod: dict, change_data: dict) -> dict:
    """
    Update permitted fields of a product dict using values from change_data.

    Only fields listed in PERMITTED_FIELDS are updated. Any other fields in
    change_data are ignored.

    Args:
        prod: The product dictionary representing the product object.
        change_data: A dictionary containing new values for fields to update.

    Returns:
        dict: The updated product dictionary (same object as input).
    """
    if not isinstance(prod, dict):
        raise TypeError("prod must be a dict")
    if not isinstance(change_data, dict):
        raise TypeError("change_data must be a dict")

    for field, value in change_data.items():
        if field in PERMITTED_FIELDS:
            prod[field] = value

    return prod
