"""
Utilities for safely updating product objects from external data sources.

This module exposes a main function, `modify_product_details`, which updates
only a predefined set of permitted fields on a product dictionary. To improve
maintainability and testability, validation, extraction of permitted updates,
and application of updates are implemented as small, pure helper functions.
"""

# Fields that are allowed to be updated from external data sources.
# Any keys not listed here will be ignored to protect product integrity.
PERMITTED_FIELDS = ["price", "description", "stock"]


def _validate_inputs(product: dict, data: dict) -> None:
    """
    Validate that `product` and `data` are dictionaries.

    Raises:
        TypeError: If either `product` or `data` is not a dict.
    """
    # Basic type checks ensure downstream code can assume dict interfaces.
    if not isinstance(product, dict):
        raise TypeError("product must be a dict")

    if not isinstance(data, dict):
        raise TypeError("data must be a dict")


def _extract_permitted_updates(data: dict, permitted_fields: list[str]) -> dict:
    """
    Build a dict containing only updates for permitted fields.

    Iterates over the whitelist of `permitted_fields` to avoid applying any
    unknown or disallowed keys that may exist in the `data` payload.

    Args:
        data (dict): Incoming updates that may contain permitted and
            non-permitted fields.
        permitted_fields (list[str]): Whitelist of fields that can be updated.

    Returns:
        dict: A new dictionary containing only permitted key/value pairs from
        `data`.
    """
    # Iterate over the whitelist to prevent accidental inclusion of
    # disallowed fields present in `data`.
    return {field: data[field] for field in permitted_fields if field in data}


def _apply_updates(product: dict, updates: dict) -> dict:
    """
    Apply updates to a product dictionary without mutating the original.

    Args:
        product (dict): The original product object.
        updates (dict): Key/value pairs to overlay on the product.

    Returns:
        dict: A new product dict with `updates` applied.
    """
    # Work on a shallow copy to preserve the input `product`.
    updated = product.copy()

    # Overlay the updates; existing keys will be replaced.
    updated.update(updates)

    return updated


def modify_product_details(product: dict, data: dict) -> dict:
    """
    Update a product dictionary with values from `data`, but only for
    fields listed in `PERMITTED_FIELDS`.

    This function is a thin orchestration layer that ensures inputs are valid,
    extracts only safe updates, and returns a new product dict without
    mutating the original.

    Args:
        product (dict): Existing product object (e.g., contains "price",
            "description", "stock", "category", etc.).
        data (dict): Incoming changes from external sources.

    Returns:
        dict: A new product dictionary with permitted fields updated.

    Raises:
        TypeError: If `product` or `data` is not a dict.
    """
    # Ensure we have the correct input types before processing.
    _validate_inputs(product, data)

    # Extract only the permitted updates (ignore any extra or disallowed keys).
    permitted_updates = _extract_permitted_updates(data, PERMITTED_FIELDS)

    # Apply the allowed updates and return the new product dict.
    return _apply_updates(product, permitted_updates)
