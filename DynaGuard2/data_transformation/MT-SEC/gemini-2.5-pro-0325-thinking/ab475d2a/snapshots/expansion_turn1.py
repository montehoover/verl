from typing import Dict

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
