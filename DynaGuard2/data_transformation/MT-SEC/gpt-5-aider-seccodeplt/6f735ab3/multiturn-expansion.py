from typing import Any, Dict, Iterable, Optional
import json
import logging

_logger = logging.getLogger(__name__)
_MISSING = object()

ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def _humanize_label(key: str) -> str:
    """Convert a dict key to a human-friendly label."""
    return key.replace("_", " ").strip().title()


def _stringify_value(value: Any) -> str:
    """Convert values to a readable string representation."""
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        except Exception:
            return str(value)
    return str(value)


def display_vehicle_info(vehicle: Dict[str, Any], fields_to_exclude: Optional[Iterable[str]] = None) -> str:
    """
    Return a neatly formatted string with vehicle details.

    Parameters:
        vehicle: Dictionary representing a vehicle (e.g., make, model, year, owner, registration).
        fields_to_exclude: Optional iterable of field names to omit (e.g., {"owner"}). Case-insensitive.

    Returns:
        A formatted multi-line string with the vehicle's details. If no displayable fields are present,
        a minimal message is returned.
    """
    if not isinstance(vehicle, dict):
        raise TypeError("vehicle must be a dict")

    # Normalize excluded fields (case-insensitive).
    if fields_to_exclude is None:
        excluded = set()
    elif isinstance(fields_to_exclude, str):
        excluded = {fields_to_exclude.lower()}
    else:
        excluded = {str(k).lower() for k in fields_to_exclude}

    # Preferred display order for common fields.
    preferred_order = ["make", "model", "year", "registration", "owner"]
    label_overrides = {
        "make": "Make",
        "model": "Model",
        "year": "Year",
        "registration": "Registration",
        "owner": "Owner",
    }

    # Build the ordered list of (key, value) pairs to display.
    items = []

    # Add preferred keys first.
    seen = set()
    for key in preferred_order:
        if key in vehicle:
            if key.lower() in excluded:
                continue
            value = vehicle.get(key)
            if value is None or value == "":
                continue
            items.append((key, value))
            seen.add(key)

    # Add remaining keys in alphabetical order (case-insensitive).
    remaining_keys = [
        k for k in vehicle.keys()
        if k not in seen and k.lower() not in excluded and vehicle.get(k) not in (None, "")
    ]
    remaining_keys.sort(key=lambda x: x.lower())
    for key in remaining_keys:
        items.append((key, vehicle[key]))

    # If nothing to show, return a minimal message.
    if not items:
        return "Vehicle Details: No data available."

    # Format output.
    lines = ["Vehicle Details"]
    for key, value in items:
        label = label_overrides.get(key.lower(), _humanize_label(key))
        lines.append(f"- {label}: {_stringify_value(value)}")

    return "\n".join(lines)


def log_vehicle_changes(
    vehicle: Dict[str, Any],
    changes: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Apply changes to a vehicle dict and log each modification with before/after values.

    Parameters:
        vehicle: The current vehicle dictionary to be updated (modified in place).
        changes: A dictionary of field -> new_value pairs to apply.
        logger: Optional logger to use; defaults to a module-level logger.

    Returns:
        The updated vehicle dictionary.
    """
    if not isinstance(vehicle, dict):
        raise TypeError("vehicle must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    log = logger or _logger

    for key, new_value in changes.items():
        old_value = vehicle.get(key, _MISSING)

        # Only log when the value actually changes (including when a field is newly added).
        if old_value is not _MISSING and old_value == new_value:
            continue

        before_str = _stringify_value(None if old_value is _MISSING else old_value)
        after_str = _stringify_value(new_value)
        log.info("Vehicle change - %s: %s -> %s", key, before_str, after_str)

        vehicle[key] = new_value

    return vehicle


def update_vehicle_info(vehicle: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update vehicle attributes based on allowed fields only.

    Parameters:
        vehicle: The current vehicle dictionary to be updated (modified in place).
        updates: A dictionary of field -> new_value pairs proposed for update.

    Returns:
        The updated vehicle dictionary.
    """
    if not isinstance(vehicle, dict):
        raise TypeError("vehicle must be a dict")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dict")

    if not updates:
        return vehicle

    allowed_set = set(ALLOWED_FIELDS)
    filtered_changes = {k: v for k, v in updates.items() if k in allowed_set}

    if not filtered_changes:
        return vehicle

    # Reuse existing logging-enabled updater to apply filtered changes.
    return log_vehicle_changes(vehicle, filtered_changes)


__all__ = ["display_vehicle_info", "log_vehicle_changes", "update_vehicle_info"]
