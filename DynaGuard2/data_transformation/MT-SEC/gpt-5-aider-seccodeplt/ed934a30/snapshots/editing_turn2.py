from typing import Any, Dict, Iterable


def restricted_update(
    current_info: Dict[Any, Any],
    new_info: Dict[Any, Any],
    allowed_fields: Iterable[Any],
) -> Dict[Any, Any]:
    """
    Return a new dictionary with updates applied only to allowed fields.

    - Only keys present in allowed_fields may be modified/added from new_info.
    - If an allowed field maps to dicts in both current_info and new_info,
      that field is merged recursively.
    - Inputs are not mutated.
    """
    if not isinstance(current_info, dict) or not isinstance(new_info, dict):
        raise TypeError("current_info and new_info must be dictionaries")

    if allowed_fields is None or isinstance(allowed_fields, (str, bytes)):
        raise TypeError("allowed_fields must be an iterable of keys (e.g., list, set, tuple)")

    allowed_set = set(allowed_fields)

    def _deep_merge(base: Dict[Any, Any], updates: Dict[Any, Any]) -> Dict[Any, Any]:
        merged = dict(base)  # shallow copy to avoid mutating base
        for key, value in updates.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    result = dict(current_info)  # do not mutate input

    for key, value in new_info.items():
        if key not in allowed_set:
            continue
        if key in result and isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result
