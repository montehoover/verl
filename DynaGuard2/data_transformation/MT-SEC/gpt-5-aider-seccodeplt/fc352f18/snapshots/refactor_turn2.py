PERMITTED_FIELDS = ["price", "description", "stock"]


def _ensure_dicts(item: dict, payload: dict) -> None:
    """
    Validate that the provided item and payload are dictionaries.
    """
    if not isinstance(item, dict):
        raise TypeError("item must be a dict")
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")


def _filter_permitted_fields(payload: dict, permitted_fields=None) -> set:
    """
    Return the set of payload keys that are permitted to be updated.
    """
    if permitted_fields is None:
        permitted_fields = PERMITTED_FIELDS
    return set(payload.keys()).intersection(permitted_fields)


def _compute_permitted_changes(item: dict, payload: dict, allowed_fields: set) -> dict:
    """
    Compute the changes to apply: only fields in allowed_fields and whose values differ.
    Returns a dict mapping field -> new_value.
    """
    changes = {}
    for field in allowed_fields:
        new_value = payload[field]
        if item.get(field) != new_value:
            changes[field] = new_value
    return changes


def amend_product_features(item: dict, payload: dict) -> dict:
    """
    Update allowed fields on a product item using values from payload.

    Only fields listed in the global PERMITTED_FIELDS are considered.
    Returns a dict of actual changes applied: {field: new_value}.
    """
    _ensure_dicts(item, payload)

    allowed_fields = _filter_permitted_fields(payload, PERMITTED_FIELDS)
    changes = _compute_permitted_changes(item, payload, allowed_fields)

    # Apply the computed changes (mutation isolated here).
    for field, value in changes.items():
        item[field] = value

    return changes
