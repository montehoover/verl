from typing import Any, Dict, Iterable, Optional
import logging


DEFAULT_FIELD_ORDER = ("make", "model", "year", "registration", "owner")


def _humanize_key(key: str) -> str:
    special = {
        "vin": "VIN",
        "id": "ID",
        "license_plate": "License Plate",
        "number_plate": "Number Plate",
    }
    lowered = key.lower()
    if lowered in special:
        return special[lowered]
    parts = lowered.replace("-", "_").split("_")
    return " ".join(p.capitalize() for p in parts if p)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes)) and value == "":
        return True
    if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
        return True
    return False


def display_vehicle_info(auto: Dict[str, Any], fields_to_hide: Optional[Iterable[str]] = None) -> str:
    """
    Return a formatted multi-line string of vehicle details.

    Parameters:
    - auto: dict with vehicle attributes (e.g., make, model, year, owner, registration, etc.)
    - fields_to_hide: iterable of keys (case-insensitive) to exclude from output, e.g., {"owner"}.

    Returns:
    - A string with one "Label: Value" per line, skipping hidden or empty fields.
    """
    if not isinstance(auto, dict):
        raise TypeError("auto must be a dict")

    hide = {k.lower() for k in (fields_to_hide or [])}
    lines: list[str] = []
    seen_keys: set[str] = set()

    # Emit known fields in a preferred order
    for key in DEFAULT_FIELD_ORDER:
        real_key = next((rk for rk in auto.keys() if rk.lower() == key), None)
        if not real_key:
            continue
        if real_key.lower() in hide:
            seen_keys.add(real_key)
            continue
        val = auto.get(real_key)
        if _is_empty(val):
            seen_keys.add(real_key)
            continue
        lines.append(f"{_humanize_key(real_key)}: {val}")
        seen_keys.add(real_key)

    # Emit any remaining fields not in the preferred order
    for rk, val in auto.items():
        if rk in seen_keys:
            continue
        if rk.lower() in hide or _is_empty(val):
            continue
        lines.append(f"{_humanize_key(rk)}: {val}")

    return "\n".join(lines)


def log_and_update_vehicle(
    auto: Dict[str, Any],
    modifications: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Update vehicle details with provided modifications and log each change.

    This function mutates the `auto` dict in-place and also returns it.

    Parameters:
    - auto: dict representing the current vehicle data.
    - modifications: dict of fields to update with their new values.
    - logger: optional logging.Logger to use; defaults to a module logger.

    Returns:
    - The updated vehicle dictionary (same object as `auto`).
    """
    if not isinstance(auto, dict):
        raise TypeError("auto must be a dict")
    if not isinstance(modifications, dict):
        raise TypeError("modifications must be a dict")

    log = logger or logging.getLogger(__name__)

    for key, new_value in modifications.items():
        old_present = key in auto
        old_value = auto.get(key)
        if (not old_present) or old_value != new_value:
            log.info("Vehicle field changed: %s: %r -> %r", key, old_value, new_value)
            auto[key] = new_value
        else:
            log.debug("Vehicle field unchanged: %s remains %r", key, old_value)

    return auto
