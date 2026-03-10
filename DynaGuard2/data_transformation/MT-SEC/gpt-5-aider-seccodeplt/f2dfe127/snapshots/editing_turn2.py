def json_to_object(data: dict) -> int:
    """
    Return the number of top-level keys in the given dictionary.

    Raises:
        TypeError: If data is not a dictionary.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")
    return len(data)
