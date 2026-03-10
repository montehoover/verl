from typing import Any, Dict

__all__ = ["update_dictionary"]

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
