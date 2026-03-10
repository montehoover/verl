PERMITTED_FIELDS = ["price", "description", "stock"]

def _filter_permitted_payload(payload: dict, permitted_fields: list) -> dict:
    """
    Filters the payload to include only keys present in permitted_fields.
    """
    return {
        key: value for key, value in payload.items() if key in permitted_fields
    }

def _apply_updates(item: dict, updates: dict) -> dict:
    """
    Applies updates to a copy of the item.
    """
    updated_item = item.copy()
    updated_item.update(updates)
    return updated_item

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
    valid_payload = _filter_permitted_payload(payload, PERMITTED_FIELDS)
    updated_item = _apply_updates(item, valid_payload)
    return updated_item
