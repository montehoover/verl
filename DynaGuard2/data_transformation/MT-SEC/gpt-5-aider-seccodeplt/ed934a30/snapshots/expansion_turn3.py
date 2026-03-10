import logging
from typing import Iterable, Mapping, Optional, Set, List, Any
from collections.abc import MutableMapping

logger = logging.getLogger(__name__)

ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def display_vehicle_info(
    details: Mapping,
    fields_to_hide: Optional[Iterable[str]] = None,
) -> str:
    """
    Return a formatted string summarizing vehicle details.

    Args:
        details: A mapping representing a vehicle, e.g. {
            "make": "Toyota",
            "model": "Camry",
            "year": 2020,
            "owner": "Jane Doe",
            "registration": "ABC-123"
        }
        fields_to_hide: Iterable of field names (case-insensitive) to exclude
            from the output, e.g. {"owner"} to hide the owner's name.

    Returns:
        A multi-line string formatted for display.
    """
    if not isinstance(details, Mapping):
        raise TypeError("details must be a mapping/dict-like object")

    hide: Set[str] = set()
    if fields_to_hide is not None:
        try:
            hide = {str(k).lower() for k in fields_to_hide}
        except TypeError as exc:
            raise TypeError("fields_to_hide must be an iterable of strings") from exc

    # Preferred display order for common vehicle fields
    preferred_order = ["make", "model", "year", "owner", "registration"]

    # Normalize keys to find matches in details while preserving original keys for display
    details_items: List[tuple[str, Any]] = []
    for k, v in details.items():
        key_str = str(k)
        details_items.append((key_str, v))

    # Determine the ordered list of keys to show
    used_lower: Set[str] = set()
    ordered_keys: List[str] = []

    # Add preferred keys that exist in details
    details_keys_lower = {k.lower(): k for k, _ in details_items}
    for key in preferred_order:
        if key.lower() in details_keys_lower:
            orig_key = details_keys_lower[key.lower()]
            if orig_key.lower() not in hide:
                ordered_keys.append(orig_key)
                used_lower.add(orig_key.lower())

    # Add remaining keys (not in preferred order), sorted alphabetically (case-insensitive)
    remaining = [
        k for k, _ in details_items
        if k.lower() not in used_lower and k.lower() not in hide
    ]
    remaining.sort(key=lambda s: s.lower())
    ordered_keys.extend(remaining)

    # Build the display lines
    lines: List[str] = ["Vehicle Information:"]
    for key in ordered_keys:
        value = details.get(key)
        if value is None:
            continue  # skip absent/None values
        label = _format_label(key)
        lines.append(f"- {label}: {value}")

    if len(lines) == 1:
        # No visible fields after filtering
        lines.append("(no visible fields)")

    return "\n".join(lines)


def log_vehicle_changes(
    details: MutableMapping,
    modifications: Mapping,
) -> MutableMapping:
    """
    Update vehicle details with the provided modifications and log each change.

    Logs:
        - Added fields
        - Updated fields (before -> after)
        - Unchanged fields at DEBUG level

    Returns:
        The same mapping instance with updates applied.
    """
    if not isinstance(details, MutableMapping):
        raise TypeError("details must be a mutable mapping/dict-like object")
    if not isinstance(modifications, Mapping):
        raise TypeError("modifications must be a mapping/dict-like object")

    _MISSING = object()

    for key, new_value in modifications.items():
        key_str = str(key)
        old_value = details.get(key, _MISSING)
        if old_value is _MISSING:
            details[key] = new_value
            logger.info("Added field %s: %r", key_str, new_value)
        elif old_value != new_value:
            logger.info("Updated field %s: %r -> %r", key_str, old_value, new_value)
            details[key] = new_value
        else:
            logger.debug("No change for field %s (value unchanged: %r)", key_str, old_value)

    return details


def alter_vehicle_info(
    details: MutableMapping,
    modifications: Mapping,
) -> MutableMapping:
    """
    Apply modifications to vehicle details but only for allowed fields.

    Only fields listed in ALLOWED_FIELDS are updated. Disallowed fields are ignored.
    Existing key casing in 'details' is preserved when possible (case-insensitive match).

    Args:
        details: Mutable mapping representing the vehicle.
        modifications: Mapping of proposed field changes.

    Returns:
        The same mapping instance with valid updates applied.
    """
    if not isinstance(details, MutableMapping):
        raise TypeError("details must be a mutable mapping/dict-like object")
    if not isinstance(modifications, Mapping):
        raise TypeError("modifications must be a mapping/dict-like object")

    allowed_lower: Set[str] = {str(f).lower() for f in ALLOWED_FIELDS}
    # Map existing keys to their original casing/object by lowercase
    existing_by_lower = {str(k).lower(): k for k in details.keys()}
    _MISSING = object()

    for key, new_value in modifications.items():
        key_str = str(key)
        key_lower = key_str.lower()

        if key_lower not in allowed_lower:
            logger.warning("Rejected modification to disallowed field %s", key_str)
            continue

        # Preserve existing key casing if present; otherwise use provided key
        target_key = existing_by_lower.get(key_lower, key)

        old_value = details.get(target_key, _MISSING)
        if old_value is _MISSING:
            details[target_key] = new_value
            logger.info("Allowed field added %s: %r", key_str, new_value)
        elif old_value != new_value:
            logger.info("Allowed field updated %s: %r -> %r", key_str, old_value, new_value)
            details[target_key] = new_value
        else:
            logger.debug("Allowed field unchanged %s (value: %r)", key_str, old_value)

    return details


def _format_label(key: str) -> str:
    """
    Convert a dictionary key into a human-friendly label.
    Preserves fully-uppercase keys (e.g., VIN), otherwise title-cases words.
    """
    if not key:
        return ""
    if key.isupper():
        return key
    # Replace common separators with spaces, then title-case
    cleaned = key.replace("_", " ").replace("-", " ").strip()
    # Special-case common acronyms within mixed-case keys
    words = cleaned.split()
    acronyms = {"vin", "id"}
    pretty_words = [
        w.upper() if w.lower() in acronyms else (w.capitalize())
        for w in words
    ]
    return " ".join(pretty_words)


__all__ = ["display_vehicle_info", "log_vehicle_changes", "alter_vehicle_info"]
