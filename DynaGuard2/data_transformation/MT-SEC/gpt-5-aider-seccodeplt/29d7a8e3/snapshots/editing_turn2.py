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


def count_data_types(data):
    """
    Count occurrences of value types in the provided dictionary.

    Returns a dictionary mapping Python type objects (e.g., int, str) to counts.
    """
    if not isinstance(data, dict):
        raise TypeError("'data' must be a dictionary")

    counts = {}
    for value in data.values():
        t = type(value)
        counts[t] = counts.get(t, 0) + 1
    return counts
