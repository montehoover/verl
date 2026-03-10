import json
from typing import Any, Dict


def read_json_content(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reads a JSON-like structure from a dictionary and returns a Python dictionary.

    Parameters:
        json_data (dict): A dictionary expected to have a 'content' key containing a JSON string.

    Returns:
        dict: The parsed dictionary if successful; otherwise, an empty dictionary.
    """
    if not isinstance(json_data, dict):
        return {}

    content = json_data.get("content")
    if not isinstance(content, str):
        return {}

    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
