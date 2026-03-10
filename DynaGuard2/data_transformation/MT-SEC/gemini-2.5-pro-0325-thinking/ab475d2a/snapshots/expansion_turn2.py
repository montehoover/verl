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

def extract_json_body(request_data: Dict[str, str]) -> Dict[Any, Any]:
    """
    Extracts and decodes JSON data from the request body.

    Args:
        request_data: A dictionary with a 'body' key containing a JSON string.

    Returns:
        The decoded JSON as a dictionary.

    Raises:
        ValueError: If the 'body' key is missing or JSON decoding fails.
    """
    if 'body' not in request_data:
        raise ValueError("Request data must contain a 'body' key.")

    try:
        return json.loads(request_data['body'])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in request body: {e}")
