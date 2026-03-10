PERMITTED_FIELDS = ["price", "description", "stock"]

def amend_product_features(item: dict, payload: dict) -> dict:
    """
    Updates certain fields of a product using incoming data.
    Only fields present in the PERMITTED_FIELDS list can be modified.

    Args:
        item: dict, the dictionary representing the product object.
        payload: dict, a dictionary containing the new values for the fields.

    Returns:
        A dictionary reflecting the changes made to the product object.
    """
    updated_item = item.copy()  # Work on a copy to avoid modifying the original dict directly if not desired
    for key, value in payload.items():
        if key in PERMITTED_FIELDS:
            updated_item[key] = value
    return updated_item
