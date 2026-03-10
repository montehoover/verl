import json
from typing import Dict, Any

def validate_content_type(headers: Dict[str, str]) -> bool:
    """
    Checks if the 'Content-Type' header is 'application/json'.

    Args:
        headers: A dictionary representing the request headers.

    Returns:
        True if 'Content-Type' is 'application/json', False otherwise.
    """
    content_type = headers.get('Content-Type')
    return content_type == 'application/json'

def extract_request_body(request: Dict) -> str:
    """
    Extracts the body from the request dictionary.

    Args:
        request: A dictionary representing the entire request.

    Returns:
        The request body as a string.
    """
    return str(request.get('body', ''))

def analyze_json_request(request: Dict) -> Dict[str, Any]:
    """
    Checks the content type of the request and parses the JSON body.

    Args:
        request: A dictionary representing the incoming request,
                 expected to have 'headers' and 'body' keys.

    Returns:
        The parsed JSON body as a dictionary.

    Raises:
        ValueError: If the content type is not 'application/json'
                    or if the JSON body is malformed.
    """
    headers = request.get('headers', {})
    if not validate_content_type(headers):
        raise ValueError("Invalid content type")

    body_str = extract_request_body(request)
    if not body_str:
        raise ValueError("Request body is empty")

    try:
        parsed_json = json.loads(body_str)
        if not isinstance(parsed_json, dict):
            # Or handle other types if necessary, but typically API JSON roots are objects
            raise ValueError("Malformed JSON: Expected a JSON object")
        return parsed_json
    except json.JSONDecodeError:
        raise ValueError("Malformed JSON")
