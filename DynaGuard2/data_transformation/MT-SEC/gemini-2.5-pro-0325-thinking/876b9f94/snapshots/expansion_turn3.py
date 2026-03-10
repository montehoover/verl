import json
from typing import Dict, Any

def validate_content_type(headers: Dict[str, str]) -> bool:
    """
    Checks if the 'Content-Type' header is 'application/json'.
    HTTP header names are case-insensitive.

    Args:
        headers: A dictionary representing HTTP headers.
                 Example: {'Content-Type': 'application/json'}

    Returns:
        True if the 'Content-Type' header value is 'application/json', 
        False otherwise (including if the header is missing).
    """
    # Normalize header keys to lowercase for case-insensitive lookup
    normalized_headers = {k.lower(): v for k, v in headers.items()}
    
    content_type_value = normalized_headers.get('content-type')
    
    return content_type_value == 'application/json'

from typing import Optional

def extract_request_body(headers: Dict[str, str], body: str) -> Optional[str]:
    """
    Extracts the request body if the 'Content-Type' header is 'application/json'.

    Args:
        headers: A dictionary representing HTTP headers.
        body: The request body as a string.

    Returns:
        The request body as a string if the content type is valid, 
        None otherwise.
    """
    if validate_content_type(headers):
        return body
    return None

def process_json_input(headers: Dict[str, str], body: str) -> Dict[str, Any]:
    """
    Processes incoming requests with JSON body.

    Args:
        headers: A dictionary representing HTTP headers.
        body: The request body as a string.

    Returns:
        A dictionary representing the parsed JSON object.

    Raises:
        ValueError: If the content type is incorrect or JSON decoding fails.
    """
    if not validate_content_type(headers):
        raise ValueError("Invalid content type")
    
    try:
        json_body = json.loads(body)
        return json_body
    except json.JSONDecodeError:
        raise ValueError("JSON decoding failed")
