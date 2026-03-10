import logging
from typing import Any, Dict, Iterable, Mapping, Tuple

ALLOWED_FIELDS = ["make", "model", "year", "registration"]

logger = logging.getLogger(__name__)
# Prevent "No handler found" warnings in library usage; applications can configure handlers/levels.
logger.addHandler(logging.NullHandler())

# Sentinel for missing keys when computing changes
_MISSING = object()


def is_allowed_field(field: str, allowed_fields: Iterable[str]) -> bool:
    """
    Determine whether a field is allowed to be updated.

    This function is pure and does not mutate any inputs.

    Args:
        field: The field name to check.
        allowed_fields: An iterable of allowed field names.

    Returns:
        True if the field is allowed, False otherwise.
    """
    # Convert to set for efficient membership test if not already a set
    allowed_set = allowed_fields if isinstance(allowed_fields, set) else set(allowed_fields)
    return field in allowed_set


def filter_allowed_modifications(
    modifications: Mapping[str, Any],
    allowed_fields: Iterable[str],
) -> Dict[str, Any]:
    """
    Return a new dict containing only modifications whose keys are allowed.

    This function is pure and does not mutate any inputs.

    Args:
        modifications: Proposed attribute updates.
        allowed_fields: Allowed field names.

    Returns:
        A new dictionary with only authorized modifications.
    """
    allowed_set = allowed_fields if isinstance(allowed_fields, set) else set(allowed_fields)
    return {k: v for k, v in modifications.items() if is_allowed_field(k, allowed_set)}


def apply_modifications_immutable(
    auto: Mapping[str, Any],
    approved_modifications: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Return a new vehicle dict with approved modifications applied.

    This function is pure and does not mutate any inputs.

    Args:
        auto: The original vehicle dictionary.
        approved_modifications: Modifications that have already been validated/filtered.

    Returns:
        A new dictionary representing the updated vehicle.
    """
    updated = dict(auto)  # shallow copy of the vehicle
    updated.update(approved_modifications)
    return updated


def _determine_attribute_changes(
    auto: Mapping[str, Any],
    approved_modifications: Mapping[str, Any],
) -> Dict[str, Tuple[Any, Any]]:
    """
    Compute the actual changes (old_value, new_value) for each approved modification.
    Only returns entries where the resulting value differs from the original or where
    the original key was not set.
    """
    changes: Dict[str, Tuple[Any, Any]] = {}
    for key, new_value in approved_modifications.items():
        old_value = auto.get(key, _MISSING)
        if old_value is _MISSING or old_value != new_value:
            changes[key] = (old_value, new_value)
    return changes


def _format_value_for_log(value: Any) -> str:
    """Format a value for human-friendly logging."""
    return "<unset>" if value is _MISSING else repr(value)


def _log_attribute_changes(changes: Mapping[str, Tuple[Any, Any]]) -> None:
    """
    Log each attribute change in a clear, readable format.

    Example:
        Updated vehicle attribute 'make': 'Ford' -> 'Tesla'
    """
    for field, (old, new) in changes.items():
        logger.info(
            "Updated vehicle attribute '%s': %s -> %s",
            field,
            _format_value_for_log(old),
            _format_value_for_log(new),
        )


def _log_ignored_modifications(
    modifications: Mapping[str, Any],
    approved_modifications: Mapping[str, Any],
) -> None:
    """Log any attempted modifications that were not allowed."""
    for field in modifications.keys():
        if field not in approved_modifications:
            logger.info("Ignored modification for disallowed attribute '%s'", field)


def adjust_vehicle_info(auto: dict, modifications: dict) -> dict:
    """
    Modify specific attributes of a vehicle dictionary based on provided inputs.
    Only attributes listed in the global ALLOWED_FIELDS can be updated.

    Logs:
        - Each updated attribute with its original and new values.
        - Any attempted modifications to disallowed attributes.

    Args:
        auto: A dictionary representing the vehicle and its attributes.
        modifications: A dictionary mapping attribute names to new values.

    Returns:
        The vehicle dictionary updated with authorized modifications applied.
    """
    if not isinstance(auto, dict):
        raise TypeError("auto must be a dict")
    if not isinstance(modifications, dict):
        raise TypeError("modifications must be a dict")

    approved = filter_allowed_modifications(modifications, ALLOWED_FIELDS)
    # Determine changes before mutating
    changes = _determine_attribute_changes(auto, approved)
    updated = apply_modifications_immutable(auto, approved)

    # Preserve original behavior by mutating the provided dict in place
    auto.clear()
    auto.update(updated)

    # Log results
    _log_attribute_changes(changes)
    _log_ignored_modifications(modifications, approved)

    return auto
