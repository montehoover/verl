"""
Utilities to amend permitted product fields safely and predictably.

Public API:
    - amend_product_features(item, payload): Apply allowed changes to a product.

Design principles:
    - Only fields listed in PERMITTED_FIELDS may be updated.
    - Validation and computation are split into small, pure helper functions to
      improve testability and maintainability.
"""

PERMITTED_FIELDS = ["price", "description", "stock"]
# Note: 'category' exists on the product object, but is intentionally excluded
# from PERMITTED_FIELDS to prevent unintended modifications.


def _ensure_dicts(item: dict, payload: dict) -> None:
    """
    Validate that the provided item and payload are dictionaries.

    Args:
        item (dict): The product object to be updated.
        payload (dict): Incoming data with potential field updates.

    Raises:
        TypeError: If either 'item' or 'payload' is not a dictionary.
    """
    if not isinstance(item, dict):
        raise TypeError("item must be a dict")
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")


def _filter_permitted_fields(payload: dict, permitted_fields=None) -> set:
    """
    Determine which keys in the payload are permitted to be updated.

    Args:
        payload (dict): Incoming data with potential field updates.
        permitted_fields (Iterable[str] | None): Collection of fields allowed
            to be updated. Defaults to PERMITTED_FIELDS if None.

    Returns:
        set: A set of payload keys that are permitted.

    Notes:
        - Using a set makes downstream operations (like intersection and
          membership checks) efficient and clear.
    """
    if permitted_fields is None:
        permitted_fields = PERMITTED_FIELDS

    # Intersect payload keys with the permitted fields to obtain the allowed set.
    return set(payload.keys()).intersection(permitted_fields)


def _compute_permitted_changes(item: dict, payload: dict, allowed_fields: set) -> dict:
    """
    Compute the concrete changes that should be applied to the item.

    Only fields present in 'allowed_fields' are considered, and only when
    the new value differs from the existing one.

    Args:
        item (dict): The current product object (source of existing values).
        payload (dict): Incoming data (source of new values).
        allowed_fields (set): The subset of payload keys permitted to change.

    Returns:
        dict: Mapping of field -> new_value for fields that will be updated.

    Examples:
        >>> item = {"price": 10, "stock": 5}
        >>> payload = {"price": 12, "stock": 5, "category": "toys"}
        >>> _compute_permitted_changes(item, payload, {"price", "stock"})
        {'price': 12}
    """
    changes = {}

    # Iterate only over permitted fields and collect differences.
    for field in allowed_fields:
        new_value = payload[field]
        # Only include a change when the value differs from the current one.
        if item.get(field) != new_value:
            changes[field] = new_value

    return changes


def amend_product_features(item: dict, payload: dict) -> dict:
    """
    Update allowed fields on a product item using values from the payload.

    This function mutates the provided 'item' in-place for permitted and
    changed fields, and returns a dictionary reflecting those applied changes.

    Args:
        item (dict): The product object to be updated. Expected keys can include
            'price', 'description', 'stock', and 'category'.
        payload (dict): New values for fields that may need updating.

    Returns:
        dict: A dict of actual changes applied: {field: new_value}.

    Raises:
        TypeError: If either 'item' or 'payload' is not a dictionary.

    Examples:
        >>> product = {"price": 100, "description": "A", "stock": 3, "category": "gadgets"}
        >>> updates = {"price": 120, "category": "premium"}
        >>> amend_product_features(product, updates)
        {'price': 120}
        >>> product
        {'price': 120, 'description': 'A', 'stock': 3, 'category': 'gadgets'}
    """
    # Validate input types early to fail fast and predictably.
    _ensure_dicts(item, payload)

    # Determine which payload fields are allowed to be updated.
    allowed_fields = _filter_permitted_fields(payload, PERMITTED_FIELDS)

    # Compute the exact changes that should be applied (pure calculation).
    changes = _compute_permitted_changes(item, payload, allowed_fields)

    # Apply the computed changes (isolated mutation for clarity and testability).
    for field, value in changes.items():
        item[field] = value

    return changes
