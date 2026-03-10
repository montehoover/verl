def update_dict_values(data: dict, updates: dict) -> dict:
    """
    Updates values in a dictionary with values from another dictionary.

    Args:
        data: The original dictionary.
        updates: A dictionary containing keys and values to update in the data.

    Returns:
        A new dictionary with the updated key-value pairs.
    """
    updated_data = data.copy()
    updated_data.update(updates)
    return updated_data


def count_data_types(data: dict) -> dict:
    """
    Counts the occurrences of each data type in the values of a dictionary.

    Args:
        data: The input dictionary.

    Returns:
        A dictionary where keys are data types and values are their counts.
    """
    type_counts = {}
    for value in data.values():
        value_type = type(value)
        type_counts[value_type] = type_counts.get(value_type, 0) + 1
    return type_counts
