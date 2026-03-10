import json
from typing import Any, Dict


def validate_content_type(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates that the input dictionary has a 'content_type' of 'application/json'
    and parses the 'content' JSON string into a Python dictionary.

    Parameters:
        json_data (dict): A dictionary expected to have:
            - 'content_type': should be 'application/json'
            - 'content': a JSON string

    Returns:
        dict: The parsed dictionary if valid and parsing succeeds; otherwise, an empty dictionary.
    """
    if not isinstance(json_data, dict):
        return {}

    if json_data.get("content_type") != "application/json":
        return {}

    content = json_data.get("content")
    if not isinstance(content, str):
        return {}

    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
