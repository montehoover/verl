import json

def json_to_obj(json_string: str):
    """
    Converts a JSON string into a Python object.

    Args:
        json_string: The JSON string to convert.

    Returns:
        The corresponding Python object.

    Raises:
        ValueError: If the input string is not valid JSON.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}") from e
