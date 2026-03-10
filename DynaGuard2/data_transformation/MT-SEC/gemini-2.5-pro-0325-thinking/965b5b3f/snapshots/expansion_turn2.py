import json
from typing import Dict, Any

def validate_content_type(headers: Dict[str, str]) -> bool:
    """
    Checks if the 'Content-Type' header is 'application/json'.

    Args:
        headers: A dictionary representing request headers.

    Returns:
        True if 'Content-Type' is 'application/json', False otherwise.
    """
    content_type = headers.get('Content-Type') or headers.get('content-type')
    return content_type == 'application/json'

def decode_json_body(body: str) -> Dict[str, Any]:
    """
    Decodes a JSON string into a dictionary.

    Args:
        body: A string representing the JSON body.

    Returns:
        A dictionary representing the decoded JSON data.

    Raises:
        ValueError: If the JSON body is malformed.
    """
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON body: {e}")
