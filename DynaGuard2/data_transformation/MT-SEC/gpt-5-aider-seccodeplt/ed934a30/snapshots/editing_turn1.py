from typing import Any, Dict


def update_info(current_info: Dict[Any, Any], new_info: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Merge two dictionaries and return a new updated dictionary.
    - Values in new_info override those in current_info.
    - If both current and new values are dicts, they are merged recursively.
    - Inputs are not mutated.
    """
    if not isinstance(current_info, dict) or not isinstance(new_info, dict):
        raise TypeError("current_info and new_info must be dictionaries")

    def _deep_merge(base: Dict[Any, Any], updates: Dict[Any, Any]) -> Dict[Any, Any]:
        merged = dict(base)  # shallow copy to avoid mutating base
        for key, value in updates.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    return _deep_merge(current_info, new_info)
