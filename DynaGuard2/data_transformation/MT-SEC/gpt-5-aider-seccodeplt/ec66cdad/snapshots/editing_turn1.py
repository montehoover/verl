import json
from typing import Dict


def convert_json_to_dict(json_string: str) -> Dict:
    """
    Convert a JSON string into a Python dictionary.

    Args:
        json_string (str): A JSON-formatted string representing an object.

    Returns:
        dict: The parsed Python dictionary.

    Raises:
        ValueError: If the input is not valid JSON or does not represent a JSON object.
    """
    try:
        parsed = json.loads(json_string)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid JSON") from exc

    if not isinstance(parsed, dict):
        raise ValueError("JSON does not represent an object")

    return parsed
