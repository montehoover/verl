import json

def convert_json_to_object(json_string: str):
    """
    Converts a JSON string into a Python object.

    Args:
        json_string: The string to convert.

    Returns:
        The corresponding Python object.

    Raises:
        ValueError: If the input string is not valid JSON.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON string") from e
