import json
from typing import Dict, Any

def validate_content_type(headers: Dict[str, str]) -> bool:
    """
    Validates if the 'Content-Type' header is 'application/json'.

    Args:
        headers: A dictionary representing request headers.
                 Header keys are typically case-insensitive.

    Returns:
        True if 'Content-Type' is 'application/json', False otherwise.
    """
    content_type = headers.get('Content-Type') or headers.get('content-type')
    return content_type == 'application/json'

def parse_json_body(body: str) -> Dict[str, Any]:
    """
    Parses a JSON string body and returns it as a dictionary.

    Args:
        body: The JSON string to parse.

    Returns:
        A dictionary representing the parsed JSON.

    Raises:
        ValueError: If the JSON body is malformed.
    """
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e}")
