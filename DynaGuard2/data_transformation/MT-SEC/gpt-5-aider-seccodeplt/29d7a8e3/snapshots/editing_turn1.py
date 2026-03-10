def update_dict_values(data, updates):
    """
    Merge two dictionaries by applying the key-value pairs from `updates` to `data`.
    Returns a new dictionary and does not mutate the inputs.
    """
    if not isinstance(data, dict) or not isinstance(updates, dict):
        raise TypeError("Both 'data' and 'updates' must be dictionaries")
    result = data.copy()
    result.update(updates)
    return result
