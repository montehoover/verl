def update_values(original: dict, new_data: dict) -> dict:
    """
    Return a new dictionary that merges 'original' with 'new_data'.

    - All keys from 'new_data' will overwrite or add to the result.
    - This performs a shallow merge and does not mutate 'original'.
    """
    if original is None:
        original = {}
    if new_data is None:
        new_data = {}

    if not isinstance(original, dict) or not isinstance(new_data, dict):
        raise TypeError("Both 'original' and 'new_data' must be dictionaries.")

    updated = original.copy()
    updated.update(new_data)
    return updated


def restricted_update(original: dict, new_data: dict, allowed_fields) -> dict:
    """
    Return a new dictionary that merges 'original' with allowed updates from 'new_data'.

    - Only keys present in 'allowed_fields' will be updated/added.
    - This performs a shallow merge and does not mutate 'original'.
    - 'allowed_fields' must be an iterable of field names (e.g., list, set, tuple).
    """
    if original is None:
        original = {}
    if new_data is None:
        new_data = {}

    if not isinstance(original, dict) or not isinstance(new_data, dict):
        raise TypeError("Both 'original' and 'new_data' must be dictionaries.")

    if allowed_fields is None:
        allowed = set()
    else:
        try:
            allowed = set(allowed_fields)
        except TypeError as exc:
            raise TypeError("`allowed_fields` must be an iterable of field names.") from exc

    updated = original.copy()
    if allowed:
        for key, value in new_data.items():
            if key in allowed:
                updated[key] = value
    return updated
