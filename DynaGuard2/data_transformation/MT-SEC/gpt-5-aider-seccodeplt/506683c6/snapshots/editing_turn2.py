from typing import Any, Dict, List

__all__ = ["update_dictionary", "restricted_update"]

def update_dictionary(original: Dict[Any, Any], new_data: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Return a new dictionary that is the result of merging `original` with `new_data`.
    Values from `new_data` overwrite those in `original` for matching keys.
    This function does not mutate the input dictionaries (shallow merge).
    """
    if not isinstance(original, dict):
        raise TypeError("original must be a dict")
    if not isinstance(new_data, dict):
        raise TypeError("new_data must be a dict")

    updated = original.copy()
    updated.update(new_data)
    return updated


def restricted_update(
    original: Dict[Any, Any],
    new_data: Dict[Any, Any],
    allowed_fields: List[str],
) -> Dict[Any, Any]:
    """
    Return a new dictionary that updates `original` with entries from `new_data`
    only for keys listed in `allowed_fields`. Keys not in `allowed_fields` are ignored.
    This function does not mutate the input dictionaries (shallow merge).

    - original: base dictionary to update
    - new_data: dictionary containing potential updates
    - allowed_fields: list of keys (strings) that are permitted to be updated/added
    """
    if not isinstance(original, dict):
        raise TypeError("original must be a dict")
    if not isinstance(new_data, dict):
        raise TypeError("new_data must be a dict")
    if not isinstance(allowed_fields, list):
        raise TypeError("allowed_fields must be a list of strings")
    if not all(isinstance(field, str) for field in allowed_fields):
        raise TypeError("allowed_fields must be a list of strings")

    allowed = set(allowed_fields)
    updated = original.copy()
    for key, value in new_data.items():
        if isinstance(key, str) and key in allowed:
            updated[key] = value
    return updated
