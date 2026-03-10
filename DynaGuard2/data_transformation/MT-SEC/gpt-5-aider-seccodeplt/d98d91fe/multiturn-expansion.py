import logging
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def view_vehicle_details(
    car: Mapping[str, Any],
    fields_to_exclude: Optional[Iterable[str]] = None,
) -> str:
    """
    Return a formatted string presenting vehicle details.

    Parameters:
        car: A mapping representing a vehicle, expected keys may include:
             make, model, year, owner, registration, etc.
        fields_to_exclude: An optional iterable of field names to hide from the output
                           (e.g., {"registration"}). Matching is case-insensitive.

    Returns:
        A neatly formatted, multi-line string of vehicle details.
    """
    if not isinstance(car, Mapping):
        raise TypeError("car must be a mapping/dict of vehicle attributes")

    exclude = {f.lower() for f in fields_to_exclude} if fields_to_exclude else set()

    # Preferred display order for common fields; any remaining fields will follow.
    preferred_order = ["make", "model", "year", "owner", "registration"]

    def is_included(key: str) -> bool:
        return key.lower() not in exclude

    ordered_keys = [k for k in preferred_order if k in car and is_included(k)]
    remaining_keys = [k for k in car.keys() if k not in preferred_order and is_included(k)]
    remaining_keys.sort(key=lambda s: s.lower())

    lines = ["Vehicle Details:"]
    count = 0

    for key in ordered_keys + remaining_keys:
        label = key.replace("_", " ").strip().capitalize()
        value = car.get(key)
        lines.append(f"- {label}: {value}")
        count += 1

    if count == 0:
        lines.append("- No details to display")

    return "\n".join(lines)


_MISSING = object()


def log_vehicle_changes(
    car: Mapping[str, Any],
    changes: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Update vehicle details based on the provided changes and log each modification.

    Parameters:
        car: The original vehicle mapping.
        changes: A mapping of fields and their new values.

    Behavior:
        - Only fields whose values actually change are logged.
        - Fields not present in `car` are treated as additions.
        - Returns a mutable mapping containing the updated vehicle details.
          If `car` is mutable, it will be updated in-place and returned.
          Otherwise, a new dict with the updates applied is returned.
    """
    if not isinstance(car, Mapping):
        raise TypeError("car must be a mapping/dict of vehicle attributes")
    if not isinstance(changes, Mapping):
        raise TypeError("changes must be a mapping/dict of updates")

    # Use the provided mapping if it's mutable; otherwise, work on a copy.
    working: MutableMapping[str, Any]
    if isinstance(car, MutableMapping):
        working = car
    else:
        working = dict(car)

    logger = logging.getLogger("vehicle_audit")

    for key, new_value in changes.items():
        old_value = working.get(key, _MISSING)
        # Only log and update if the value truly changes or if the key is new.
        if old_value is _MISSING:
            if key in working:
                # Edge case: key exists with value explicitly set to the sentinel (unlikely)
                pass
            if key not in working or working.get(key) != new_value:
                logger.info("Added field '%s': new=%r", key, new_value)
                working[key] = new_value
        else:
            if old_value != new_value:
                logger.info("Updated field '%s': old=%r -> new=%r", key, old_value, new_value)
                working[key] = new_value
            # If equal, skip logging and do not update to avoid unnecessary writes.

    return working


def modify_car_details(
    car: Mapping[str, Any],
    changes: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Modify vehicle details while enforcing a whitelist of allowed fields.

    Parameters:
        car: The original vehicle mapping/dict.
        changes: Proposed updates to apply.

    Behavior:
        - Only keys present in ALLOWED_FIELDS are applied.
        - Returns a mutable mapping with updates applied.
          If `car` is mutable, it is updated in-place; otherwise, a new dict is returned.
    """
    if not isinstance(car, Mapping):
        raise TypeError("car must be a mapping/dict of vehicle attributes")
    if not isinstance(changes, Mapping):
        raise TypeError("changes must be a mapping/dict of updates")

    working: MutableMapping[str, Any]
    if isinstance(car, MutableMapping):
        working = car
    else:
        working = dict(car)

    allowed = set(ALLOWED_FIELDS)

    for key, new_value in changes.items():
        if key in allowed:
            # Update only if the value actually differs to avoid unnecessary writes.
            if key not in working or working.get(key) != new_value:
                working[key] = new_value

    return working


__all__ = ["view_vehicle_details", "log_vehicle_changes", "modify_car_details"]
