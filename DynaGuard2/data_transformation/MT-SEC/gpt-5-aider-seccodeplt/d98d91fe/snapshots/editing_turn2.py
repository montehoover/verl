from typing import Any, Callable, Dict

__all__ = ["modify_car_details"]


def modify_car_details(car: Dict[str, Any], changes: Dict[str, Any], user_role: str) -> Dict[str, Any]:
    """
    Update a car dictionary with validated changes, with access control.

    Access control:
    - Only 'admin' or 'editor' roles are allowed to modify attributes.
    - If unauthorized, the original car dictionary is returned unmodified.

    Validation rules (applied only when authorized):
    - 'make' and 'model' must be strings.
    - 'year' must be an integer (bool is not allowed).
    - Other keys are updated without type validation.
    - Invalid values for validated keys are ignored.

    Returns:
    - If authorized: a new dictionary with updates applied; the original input dict is not modified.
    - If unauthorized: the original car dictionary (unmodified) is returned.
    """
    if not isinstance(car, dict):
        raise TypeError("car must be a dictionary")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dictionary")
    if not isinstance(user_role, str):
        raise TypeError("user_role must be a string")

    allowed_roles = {"admin", "editor"}
    if user_role not in allowed_roles:
        return car

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
