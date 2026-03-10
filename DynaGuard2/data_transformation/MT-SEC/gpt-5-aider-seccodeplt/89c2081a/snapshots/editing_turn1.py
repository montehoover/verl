import json
from typing import Dict, Any


def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    Args:
        json_string: A string containing a JSON object.

    Returns:
        A Python dictionary parsed from the JSON string.

    Raises:
        ValueError: If the input is not valid JSON or does not represent a JSON object.
        TypeError: If json_string is not a string.
    """
    if not isinstance(json_string, str):
        raise TypeError("json_string must be a string")

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON") from e

    if not isinstance(data, dict):
        raise ValueError("JSON must represent an object")

    return data
