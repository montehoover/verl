from typing import Dict

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
