import json
from typing import Any, Dict

def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    :param json_string: JSON string representing an object.
    :return: The parsed dictionary.
    :raises ValueError: If the input isn't valid JSON or doesn't represent a JSON object.
    :raises TypeError: If json_string is not a string.
    """
    if not isinstance(json_string, str):
        raise TypeError("json_string must be a string")

    try:
        result = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg}") from e

    if not isinstance(result, dict):
        raise ValueError("JSON must represent an object (dictionary)")

    return result
