"""
Utilities for updating vehicle details with strict control over which fields
are allowed to be modified. This module provides pure helper functions to
determine allowed fields, filter changes, and apply updates. Logging is used
to trace updates and decisions made during the update process.
"""

import logging

ALLOWED_FIELDS = ["make", "model", "year", "registration"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def is_allowed_field(field: str) -> bool:
    """
    Determine if a field is allowed to be updated.

    Args:
        field (str): The field name to check.

    Returns:
        bool: True if the field is allowed, False otherwise.
    """
    allowed = field in ALLOWED_FIELDS
    logger.debug("Field '%s' allowed: %s", field, allowed)
    return allowed


def filter_allowed_changes(changes: dict) -> dict:
    """
    Filter the provided changes to include only allowed fields.

    Args:
        changes (dict): Proposed field updates.

    Returns:
        dict: A new dictionary containing only allowed updates.
    """
    allowed_changes = {}
    for field, value in changes.items():
        if is_allowed_field(field):
            allowed_changes[field] = value
        else:
            logger.debug("Ignoring disallowed field '%s' with value: %r", field, value)
    logger.debug("Filtered allowed changes: %r", allowed_changes)
    return allowed_changes


def apply_updates(car_details: dict, allowed_changes: dict) -> dict:
    """
    Apply allowed changes to the car details without mutating inputs.

    Args:
        car_details (dict): Original vehicle details.
        allowed_changes (dict): Validated changes that are allowed.

    Returns:
        dict: A new dictionary with the allowed changes applied.
    """
    updated = dict(car_details)
    for field, new_value in allowed_changes.items():
        old_value = updated.get(field)
        logger.info(
            "Updating field '%s': %r -> %r",
            field,
            old_value,
            new_value,
        )
        updated[field] = new_value
    return updated


def update_car_information(car_details: dict, changes: dict) -> dict:
    """
    Orchestrate the update process for a vehicle's details.

    Filters the proposed changes to include only allowed fields and applies
    them to a copy of the original details, ensuring inputs are not mutated.

    Args:
        car_details (dict): Original vehicle details.
        changes (dict): Proposed field updates.

    Returns:
        dict: A new dictionary with allowed updates applied.
    """
    logger.debug("Received change request: %r", changes)
    allowed_changes = filter_allowed_changes(changes)

    if not allowed_changes:
        logger.info("No allowed changes to apply.")
        return dict(car_details)

    updated = apply_updates(car_details, allowed_changes)
    logger.debug("Update result: %r", updated)
    return updated
