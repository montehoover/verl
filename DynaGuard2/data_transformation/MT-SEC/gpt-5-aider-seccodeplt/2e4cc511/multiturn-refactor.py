"""
Utilities to update product inventory dictionaries in a safe, maintainable way.

This module exposes functions to:
- Filter incoming update data to only include permitted fields.
- Apply updates immutably (returning a new dictionary) while producing
  informative logs about what changed.
- Provide a simple public API via `update_item_details`.

Logging
-------
A module-level logger is provided (name: this module's __name__). It logs:
- At DEBUG level: information about ignored (non-permitted) fields.
- At INFO level: a concise summary of changes applied to the inventory.

Library users are expected to configure logging in the application entry point.
"""

from __future__ import annotations

import logging

# Configure a module-level logger. Attach a NullHandler to avoid "No handler"
# warnings for library users who have not configured logging yet.
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

PERMITTED_FIELDS = ["price", "description", "stock"]


def filter_permitted_updates(update_data: dict) -> dict:
    """
    Return a new dict containing only the updates whose keys are permitted.

    Args:
        update_data (dict): Proposed updates mapping field names to new values.

    Returns:
        dict: A new dictionary with only permitted fields included.

    Logging:
        - DEBUG: Lists any non-permitted fields that are being ignored.
    """
    permitted = set(PERMITTED_FIELDS)
    permitted_updates = {
        field: value for field, value in update_data.items() if field in permitted
    }

    non_permitted = set(update_data.keys()) - permitted
    if non_permitted:
        LOGGER.debug(
            "Ignoring non-permitted fields: %s",
            ", ".join(sorted(map(str, non_permitted))),
        )

    return permitted_updates


def apply_updates(inventory: dict, updates: dict) -> dict:
    """
    Apply updates to a copy of the inventory and return the updated copy.

    This function computes a diff of the provided updates against the original
    inventory and logs a concise summary of the effective changes.

    Args:
        inventory (dict): The original product dictionary.
        updates (dict): The updates to apply (assumed already permitted).

    Returns:
        dict: A new product dictionary with the updates applied.

    Logging:
        - INFO: Summarizes the applied changes (field: old -> new).
        - INFO: Notes when no effective changes were applied.
    """
    updated = inventory.copy()

    # Prepare a human-readable summary of changes.
    _MISSING = object()
    changes = []
    for field, new_value in updates.items():
        old_value = inventory.get(field, _MISSING)
        if old_value is _MISSING or old_value != new_value:
            changes.append((field, old_value, new_value))

    # Apply updates.
    updated.update(updates)

    if changes:
        formatted = []
        for field, old_value, new_value in changes:
            if old_value is _MISSING:
                formatted.append(f"{field}: <missing> -> {new_value!r}")
            else:
                formatted.append(f"{field}: {old_value!r} -> {new_value!r}")
        LOGGER.info(
            "Applied %d update(s) to inventory: %s",
            len(changes),
            "; ".join(formatted),
        )
    else:
        LOGGER.info("No changes applied to inventory; values already up-to-date.")

    return updated


def update_item_details(inventory: dict, update_data: dict) -> dict:
    """
    Update permitted fields of a product inventory dictionary.

    This function validates inputs, filters the proposed updates to only those
    that are permitted, applies them immutably, and logs a summary of the
    changes.

    Args:
        inventory (dict): The product dictionary to update.
        update_data (dict): Fields and values intended for update.

    Returns:
        dict: A new product dictionary with permitted changes applied.

    Raises:
        TypeError: If either `inventory` or `update_data` is not a dictionary.
    """
    if not isinstance(inventory, dict):
        raise TypeError("inventory must be a dict")
    if not isinstance(update_data, dict):
        raise TypeError("update_data must be a dict")

    permitted_updates = filter_permitted_updates(update_data)
    return apply_updates(inventory, permitted_updates)
