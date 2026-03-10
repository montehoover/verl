import logging

ALLOWED_FIELDS = ["make", "model", "year", "registration"]

# Module-level logger (library-friendly: no configuration here)
logger = logging.getLogger(__name__)

_MISSING = object()


def is_allowed_field(field: str, allowed_fields=None) -> bool:
    """
    Check if a given field is allowed to be modified.

    Args:
        field: The field name to check.
        allowed_fields: An optional iterable of allowed field names. Defaults to ALLOWED_FIELDS.

    Returns:
        True if the field is allowed; otherwise, False.
    """
    if allowed_fields is None:
        allowed_fields = ALLOWED_FIELDS
    return field in allowed_fields


def _repr_value(value) -> str:
    """Represent a value for logging, handling the 'missing' sentinel."""
    if value is _MISSING:
        return "<missing>"
    return repr(value)


def _log_attribute_change(field: str, old_value, new_value, action: str, log: logging.Logger = None) -> None:
    """
    Log a single attribute change with original and updated values.

    Args:
        field: The attribute name being changed.
        old_value: The original value, or _MISSING if it did not exist.
        new_value: The new value being set.
        action: A short action label: 'added' or 'updated'.
        log: Optional logger to use; defaults to module logger.
    """
    if log is None:
        log = logger
    log.info(
        "Car attribute %s %s: %s -> %s",
        field,
        action,
        _repr_value(old_value),
        _repr_value(new_value),
    )


def apply_allowed_updates(car: dict, updates: dict, allowed_fields=None) -> dict:
    """
    Apply updates to a car dictionary, only modifying fields that are allowed.
    Logs each modification with the original and updated values.

    Args:
        car: Current state of vehicle attributes.
        updates: Fields and new values to update.
        allowed_fields: An optional iterable of allowed field names. Defaults to ALLOWED_FIELDS.

    Returns:
        A new car dictionary with allowed updates applied.
    """
    if allowed_fields is None:
        allowed_fields = ALLOWED_FIELDS

    updated_car = dict(car)  # do not mutate the original input

    for key, value in updates.items():
        if is_allowed_field(key, allowed_fields):
            previous = updated_car.get(key, _MISSING)
            # Only log and apply when a real modification occurs
            if previous is _MISSING:
                updated_car[key] = value
                _log_attribute_change(key, previous, value, action="added")
            elif previous != value:
                updated_car[key] = value
                _log_attribute_change(key, previous, value, action="updated")
            else:
                # No actual change; useful debug trace
                logger.debug("No change for attribute %s (value unchanged: %s)", key, _repr_value(value))
        else:
            # Field not allowed; useful for debugging why an update didn't apply
            logger.debug("Ignored update for disallowed attribute %s", key)

    return updated_car


def modify_car_attributes(car: dict, updates: dict) -> dict:
    """
    Update specific fields of a vehicle's attributes based on the ALLOWED_FIELDS constraint.
    Logs every modification with original and updated values.

    Args:
        car: Current state of vehicle attributes.
        updates: Fields and new values to update.

    Returns:
        The updated vehicle dictionary reflecting any valid modifications.
    """
    return apply_allowed_updates(car, updates)
