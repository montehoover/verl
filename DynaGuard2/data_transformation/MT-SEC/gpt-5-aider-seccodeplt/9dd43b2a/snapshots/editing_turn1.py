import json
from typing import Any, Dict


def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    :param json_string: The JSON string to parse.
    :return: The parsed dictionary.
    :raises ValueError: If the input is not valid JSON or does not represent a JSON object.
    """
    try:
        parsed = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg}") from e

    if not isinstance(parsed, dict):
        raise ValueError("JSON must represent an object at the top level")

    return parsed
