import json
from typing import Any, Dict

def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    Args:
        json_string: A JSON-encoded string representing an object.

    Returns:
        A Python dictionary parsed from the JSON string.

    Raises:
        ValueError: If the input is not valid JSON or does not represent a JSON object.
    """
    try:
        result = json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Invalid JSON input") from exc

    if not isinstance(result, dict):
        raise ValueError("JSON input must represent an object")

    return result
