def json_to_object(data: dict):
    """
    Measures how many keys are in a given dictionary and returns that count.

    Args:
        data: A dictionary.

    Returns:
        The number of top-level keys in the dictionary.

    Raises:
        TypeError: If the input is not a dictionary.
    """
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary.")
    return len(data)
