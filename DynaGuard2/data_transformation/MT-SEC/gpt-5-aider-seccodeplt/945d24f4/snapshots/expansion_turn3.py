import logging
from typing import Iterable, Mapping, MutableMapping, Optional


_logger = logging.getLogger(__name__)

ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def display_car_details(car_details: Mapping[str, object], exclude: Optional[Iterable[str]] = None) -> str:
    """
    Return a formatted string of car details.

    Parameters:
        car_details: A mapping containing vehicle properties (e.g., make, model, year, owner, registration).
        exclude: Optional iterable of field names to omit from the output (case-insensitive), e.g., {"owner"}.

    Returns:
        A newline-separated string with labels and values, ordered with common fields first.
    """
    if not isinstance(car_details, Mapping):
        raise TypeError("car_details must be a mapping/dict of field names to values")

    exclude_set = {str(e).lower() for e in exclude} if exclude else set()

    preferred_order = ["make", "model", "year", "owner", "registration"]

    def is_excluded(key: str) -> bool:
        return key.lower() in exclude_set

    # Build an ordered list of keys: preferred first (if present), then remaining alphabetical.
    preferred_keys = [k for k in preferred_order if k in car_details and not is_excluded(k)]
    remaining_keys = [k for k in car_details.keys() if k not in preferred_order and not is_excluded(k)]
    remaining_keys.sort(key=lambda s: s.lower())

    ordered_keys = preferred_keys + remaining_keys

    lines = []
    for key in ordered_keys:
        value = car_details.get(key)
        label = key.replace("_", " ").strip().title()
        if value is None or (isinstance(value, str) and value.strip() == ""):
            value_str = "N/A"
        else:
            value_str = str(value)
        lines.append(f"{label}: {value_str}")

    return "\n".join(lines)


_MISSING = object()


def log_and_update_car(
    car_details: MutableMapping[str, object],
    changes: Mapping[str, object],
    logger: Optional[logging.Logger] = None,
) -> MutableMapping[str, object]:
    """
    Update car details in-place while logging each change.

    Parameters:
        car_details: A mutable mapping of the current vehicle details to update.
        changes: A mapping of fields to new values.
        logger: Optional logger to use; defaults to a module-level logger.

    Behavior:
        - For each (key, value) in `changes`, set car_details[key] = value.
        - Log only when a value actually changes or a new key is added.
        - The log entry includes the field name and the before/after values.

    Returns:
        The same `car_details` mapping after applying updates.
    """
    if not isinstance(car_details, MutableMapping):
        raise TypeError("car_details must be a mutable mapping/dict to apply updates")
    if not isinstance(changes, Mapping):
        raise TypeError("changes must be a mapping/dict of field names to new values")

    log = logger or _logger

    for key, new_value in changes.items():
        old_value = car_details.get(key, _MISSING)

        # Only log when there's an actual change (including new fields).
        if old_value is _MISSING or old_value != new_value:
            before_repr = "<missing>" if old_value is _MISSING else repr(old_value)
            after_repr = repr(new_value)
            action = "Added" if old_value is _MISSING else "Updated"
            log.info("%s field '%s': %s -> %s", action, key, before_repr, after_repr)
            car_details[key] = new_value

    return car_details


def update_car_information(
    car_details: MutableMapping[str, object],
    changes: Mapping[str, object],
) -> MutableMapping[str, object]:
    """
    Update car details in-place but only for fields allowed by ALLOWED_FIELDS.

    Parameters:
        car_details: A mutable mapping of current vehicle details.
        changes: A mapping of proposed updates.

    Returns:
        The same `car_details` mapping after applying allowed updates.
    """
    if not isinstance(car_details, MutableMapping):
        raise TypeError("car_details must be a mutable mapping/dict to apply updates")
    if not isinstance(changes, Mapping):
        raise TypeError("changes must be a mapping/dict of field names to new values")

    for key, value in changes.items():
        if key in ALLOWED_FIELDS:
            car_details[key] = value

    return car_details
