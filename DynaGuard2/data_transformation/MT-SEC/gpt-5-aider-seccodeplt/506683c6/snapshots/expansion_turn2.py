from typing import Any, Dict, Iterable, Optional, Set, List
import logging

_LOGGER = logging.getLogger(__name__)


def display_vehicle_info(
    car: Dict[str, Any],
    exclude: Optional[Iterable[str]] = None
) -> str:
    """
    Return a formatted string of vehicle details.

    Parameters:
        car: A dictionary representing a vehicle. Expected keys may include:
             'make', 'model', 'year', 'owner', 'registration', and any others.
        exclude: An optional iterable of keys to omit from the output (e.g., {'owner'}).

    Returns:
        A human-readable string presenting the vehicle details.
    """
    if not isinstance(car, dict):
        raise TypeError("car must be a dict")

    # Normalize exclude set
    excluded: Set[str] = set(exclude or [])

    # Preferred display order for common fields
    preferred_order: List[str] = ["make", "model", "year", "owner", "registration"]

    # Label mapping for nicer display
    labels: Dict[str, str] = {
        "make": "Make",
        "model": "Model",
        "year": "Year",
        "owner": "Owner",
        "registration": "Registration",
    }

    # Determine the full ordered list of keys to display:
    # 1) Preferred ordered keys that exist in car and aren't excluded
    ordered_keys: List[str] = [
        k for k in preferred_order if k in car and k not in excluded
    ]

    # 2) Any remaining keys (not already included), sorted alphabetically
    remaining_keys = sorted(
        k for k in car.keys() if k not in excluded and k not in ordered_keys
    )
    ordered_keys.extend(remaining_keys)

    # Helper to format values consistently
    def _format_value(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, str) and value.strip() == "":
            return "N/A"
        return str(value)

    lines = ["Vehicle Information:"]
    for key in ordered_keys:
        label = labels.get(key, key.replace("_", " ").title())
        value = _format_value(car.get(key))
        lines.append(f"- {label}: {value}")

    return "\n".join(lines)


def log_vehicle_changes(
    car: Dict[str, Any],
    updates: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Apply updates to a vehicle dictionary and log each change.

    Parameters:
        car: The vehicle dictionary to update.
        updates: A dictionary of updates to apply to the vehicle.
        logger: Optional logger to use. Defaults to a module-level logger.

    Returns:
        The updated vehicle dictionary (the same object, modified in place).
    """
    if not isinstance(car, dict):
        raise TypeError("car must be a dict")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dict")

    log = logger or _LOGGER
    _MISSING = object()

    for key, new_value in updates.items():
        old_value = car.get(key, _MISSING)
        if old_value is _MISSING or old_value != new_value:
            old_for_log = "<missing>" if old_value is _MISSING else old_value
            log.info("Vehicle update - %s: %r -> %r", key, old_for_log, new_value)
            car[key] = new_value

    return car
