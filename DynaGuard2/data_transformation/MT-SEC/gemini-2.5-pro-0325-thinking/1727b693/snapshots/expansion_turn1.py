from typing import Dict

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
