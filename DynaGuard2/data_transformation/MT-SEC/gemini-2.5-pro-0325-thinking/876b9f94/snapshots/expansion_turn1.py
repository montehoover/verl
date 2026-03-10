from typing import Dict

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
