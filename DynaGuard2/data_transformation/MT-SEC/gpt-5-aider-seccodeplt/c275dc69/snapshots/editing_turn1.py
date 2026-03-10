import json
from typing import Dict, Any


def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    Args:
        json_string: A JSON-formatted string representing an object.

    Returns:
        A dictionary parsed from the JSON string.

    Raises:
        ValueError: If the input is not a string, not valid JSON, or does not represent a JSON object.
    """
    if not isinstance(json_string, str):
        raise ValueError("json_string must be a string")

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg}") from e

    if not isinstance(data, dict):
        raise ValueError("JSON must represent an object")

    return data
