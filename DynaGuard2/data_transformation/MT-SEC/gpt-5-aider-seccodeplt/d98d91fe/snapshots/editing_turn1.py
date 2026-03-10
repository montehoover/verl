from typing import Any, Callable, Dict

__all__ = ["modify_car_details"]


def modify_car_details(car: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a car dictionary with validated changes.

    Validation rules:
    - 'make' and 'model' must be strings.
    - 'year' must be an integer (bool is not allowed).
    - Other keys are updated without type validation.
    - Invalid values for validated keys are ignored.

    Returns a new dictionary with updates applied; the original input dict is not modified.
    """
    if not isinstance(car, dict):
        raise TypeError("car must be a dictionary")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dictionary")

    def _is_str(v: Any) -> bool:
        return isinstance(v, str)

    def _is_int_not_bool(v: Any) -> bool:
        # In Python, bool is a subclass of int; explicitly exclude it.
        return isinstance(v, int) and not isinstance(v, bool)

    validators: Dict[str, Callable[[Any], bool]] = {
        "make": _is_str,
        "model": _is_str,
        "year": _is_int_not_bool,
    }

    updated: Dict[str, Any] = dict(car)

    for key, value in changes.items():
        if key in validators:
            if validators[key](value):
                updated[key] = value
            else:
                # Skip invalid type for validated keys
                continue
        else:
            # Apply non-validated keys as-is
            updated[key] = value

    return updated
