import json
from typing import Any

def convert_json_to_object(json_string: str) -> Any:
    """
    Convert a JSON string to the corresponding Python object.

    Args:
        json_string (str): A string containing JSON.

    Returns:
        Any: The Python object represented by the JSON string.

    Raises:
        ValueError: If the input is not valid JSON.
        TypeError: If json_string is not a string.
    """
    if not isinstance(json_string, str):
        raise TypeError("json_string must be a string")

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(str(e)) from e
