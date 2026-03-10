import json
from typing import Any, Dict

def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    Args:
        json_string: A string containing JSON that represents an object.

    Returns:
        A dictionary parsed from the JSON string.

    Raises:
        ValueError: If the input is not valid JSON or does not represent a JSON object.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON") from e

    if not isinstance(data, dict):
        raise ValueError("JSON does not represent an object")

    return data
