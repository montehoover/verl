from typing import Any, Callable, Dict

ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def _pure_update_field(car_data: Dict[str, Any], field_name: str, new_value: Any) -> Dict[str, Any]:
    """
    Purely creates a new dictionary with a specified field updated.

    This function ensures that the original dictionary is not modified by
    creating a shallow copy before applying the update.

    Args:
        car_data: The original dictionary representing car data.
        field_name: The name of the field (attribute) to update.
        new_value: The new value for the specified field.

    Returns:
        A new dictionary with the specified field updated.
    """
    updated_data = car_data.copy()
    updated_data[field_name] = new_value
    return updated_data


def _update_car_make(car_data: Dict[str, Any], new_value: Any) -> Dict[str, Any]:
    """
    Purely updates the 'make' attribute of a car data dictionary.

    Creates a new dictionary with the 'make' updated, leaving the original
    dictionary unchanged.

    Args:
        car_data: The car data dictionary.
        new_value: The new value for the 'make' attribute.

    Returns:
        A new dictionary with the 'make' attribute updated.
    """
    return _pure_update_field(car_data, "make", new_value)


def _update_car_model(car_data: Dict[str, Any], new_value: Any) -> Dict[str, Any]:
    """
    Purely updates the 'model' attribute of a car data dictionary.

    Creates a new dictionary with the 'model' updated, leaving the original
    dictionary unchanged.

    Args:
        car_data: The car data dictionary.
        new_value: The new value for the 'model' attribute.

    Returns:
        A new dictionary with the 'model' attribute updated.
    """
    return _pure_update_field(car_data, "model", new_value)


def _update_car_year(car_data: Dict[str, Any], new_value: Any) -> Dict[str, Any]:
    """
    Purely updates the 'year' attribute of a car data dictionary.

    Creates a new dictionary with the 'year' updated, leaving the original
    dictionary unchanged.

    Args:
        car_data: The car data dictionary.
        new_value: The new value for the 'year' attribute.

    Returns:
        A new dictionary with the 'year' attribute updated.
    """
    return _pure_update_field(car_data, "year", new_value)


def _update_car_registration(car_data: Dict[str, Any], new_value: Any) -> Dict[str, Any]:
    """
    Purely updates the 'registration' attribute of a car data dictionary.

    Creates a new dictionary with the 'registration' updated, leaving the original
    dictionary unchanged.

    Args:
        car_data: The car data dictionary.
        new_value: The new value for the 'registration' attribute.

    Returns:
        A new dictionary with the 'registration' attribute updated.
    """
    return _pure_update_field(car_data, "registration", new_value)


_ATTRIBUTE_UPDATERS: Dict[str, Callable[[Dict[str, Any], Any], Dict[str, Any]]] = {
    "make": _update_car_make,
    "model": _update_car_model,
    "year": _update_car_year,
    "registration": _update_car_registration,
}

def modify_car_details(car: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modifies specific attributes of a vehicle object based on provided inputs.
    Uses pure functions for attribute updates and guard clauses for early exit.
    If no valid or effective changes are provided, the original car object is returned.

    Args:
        car: A dictionary object representing the vehicle with its attributes.
        changes: A dictionary of the new values mapped to the attributes
                 that need updating.

    Returns:
        The updated vehicle dictionary if modifications were made, otherwise the
        original car dictionary.
    """
    # Guard clause: if no changes dictionary is provided or it's empty.
    if not changes:
        return car

    # Filter changes to only include allowed fields and actual modifications.
    # An actual modification means the new value is different from the current one.
    relevant_changes_to_apply = {}
    for key, value in changes.items():
        if key in ALLOWED_FIELDS and car.get(key) != value:
            relevant_changes_to_apply[key] = value

    # Guard clause: if no relevant or allowed changes after filtering,
    # return the original car object.
    if not relevant_changes_to_apply:
        return car

    # If we reach here, modifications are needed.
    # Start with the original car reference. Each pure update function
    # will return a new dictionary instance, ensuring purity.
    updated_car_state = car
    for key, value in relevant_changes_to_apply.items():
        updater_func = _ATTRIBUTE_UPDATERS.get(key)
        # This check is technically redundant if ALLOWED_FIELDS and _ATTRIBUTE_UPDATERS
        # are in sync, but good for robustness.
        if updater_func:
            updated_car_state = updater_func(updated_car_state, value)

    return updated_car_state
