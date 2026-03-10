from typing import Any, Dict, Iterable, Mapping

ALLOWED_FIELDS = ["make", "model", "year", "registration"]


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


def adjust_vehicle_info(auto: dict, modifications: dict) -> dict:
    """
    Modify specific attributes of a vehicle dictionary based on provided inputs.
    Only attributes listed in the global ALLOWED_FIELDS can be updated.

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
    updated = apply_modifications_immutable(auto, approved)

    # Preserve original behavior by mutating the provided dict in place
    auto.clear()
    auto.update(updated)

    return auto
