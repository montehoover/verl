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
