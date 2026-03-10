import json
from typing import Any, Dict

def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    Args:
        json_string (str): A string containing a JSON object.

    Returns:
        Dict[str, Any]: The parsed dictionary.

    Raises:
        ValueError: If the input is not valid JSON or does not represent a JSON object.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg}") from e

    if not isinstance(data, dict):
        raise ValueError("JSON must represent an object")

    return data
