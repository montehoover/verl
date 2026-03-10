PERMITTED_FIELDS = ["price", "description", "stock"]


def amend_product_features(item: dict, payload: dict) -> dict:
    """
    Update allowed fields on a product item using values from payload.

    Only fields listed in the global PERMITTED_FIELDS are considered.
    Returns a dict of actual changes applied: {field: new_value}.
    """
    if not isinstance(item, dict):
        raise TypeError("item must be a dict")
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    changes = {}

    for field in PERMITTED_FIELDS:
        if field in payload:
            new_value = payload[field]
            if item.get(field) != new_value:
                item[field] = new_value
                changes[field] = new_value

    return changes
