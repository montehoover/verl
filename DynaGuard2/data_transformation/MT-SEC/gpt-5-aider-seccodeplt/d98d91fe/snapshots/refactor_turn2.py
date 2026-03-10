ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def _updated_make(current_car: dict, new_value):
    """Pure function to compute updated 'make' value."""
    return new_value


def _updated_model(current_car: dict, new_value):
    """Pure function to compute updated 'model' value."""
    return new_value


def _updated_year(current_car: dict, new_value):
    """Pure function to compute updated 'year' value."""
    return new_value


def _updated_registration(current_car: dict, new_value):
    """Pure function to compute updated 'registration' value."""
    return new_value


# Map each allowed field to its pure update function
_VALUE_UPDATERS = {
    "make": _updated_make,
    "model": _updated_model,
    "year": _updated_year,
    "registration": _updated_registration,
}


def modify_car_details(car: dict, changes: dict) -> dict:
    """
    Modify specific attributes of a vehicle dictionary based on provided inputs.

    Only fields listed in ALLOWED_FIELDS are updated. Fields not listed are ignored.
    The original car dictionary is not mutated; a new updated dictionary is returned.

    Guard clauses are used to early-exit when there are no applicable modifications.

    Args:
        car: dict representing the vehicle and its attributes.
        changes: dict mapping attribute names to new values.

    Returns:
        dict: Updated vehicle dictionary reflecting authorized modifications.
    """
    if not isinstance(car, dict):
        raise TypeError("car must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    # Early exit: no changes provided
    if not changes:
        return dict(car)

    # Filter to only allowed fields
    allowed_change_keys = [k for k in changes.keys() if k in ALLOWED_FIELDS]
    if not allowed_change_keys:
        return dict(car)

    # Determine which allowed fields actually change values
    fields_to_modify = [
        k for k in allowed_change_keys
        if car.get(k) != changes[k]
    ]
    if not fields_to_modify:
        return dict(car)

    # Apply updates via pure per-field updaters
    updated_car = dict(car)
    for field in fields_to_modify:
        updater = _VALUE_UPDATERS.get(field)
        new_value = changes[field]
        if updater is not None:
            updated_car[field] = updater(updated_car, new_value)
        else:
            # Fallback: should not happen if ALLOWED_FIELDS and _VALUE_UPDATERS are in sync
            updated_car[field] = new_value

    return updated_car
