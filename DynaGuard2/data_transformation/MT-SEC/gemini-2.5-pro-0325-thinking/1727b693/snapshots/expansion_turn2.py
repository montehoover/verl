import json
from typing import Dict, Any

def check_header_for_json(headers: Dict[str, str]) -> bool:
    """
    Checks if the Content-Type header is set to 'application/json'.

    Args:
        headers: A dictionary representing request headers.

    Returns:
        True if Content-Type is 'application/json', False otherwise.
    """
    content_type = headers.get("Content-Type")
    if content_type and content_type.lower() == "application/json":
        return True
    return False

def decode_json_body(json_string: str) -> Dict[Any, Any]:
    """
    Decodes a JSON-encoded string into a Python dictionary.

    Args:
        json_string: The JSON-encoded string.

    Returns:
        A dictionary representing the parsed JSON data.

    Raises:
        ValueError: If the JSON string is improperly formatted.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
