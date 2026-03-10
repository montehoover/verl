def check_content_type(headers: dict) -> bool:
    """
    Verifies if the content type of an HTTP request is 'application/json'.

    Args:
        headers: A dictionary containing HTTP headers.

    Returns:
        True if the content type is 'application/json', otherwise False.
    """
    content_type = headers.get('Content-Type')
    if content_type == 'application/json':
        return True
    return False
