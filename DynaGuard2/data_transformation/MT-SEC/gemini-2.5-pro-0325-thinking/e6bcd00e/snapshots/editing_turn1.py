def check_content_type(request_headers: dict) -> bool:
    """
    Verifies if the content type of an incoming request is 'application/json'.

    Args:
        request_headers: A dictionary containing the HTTP headers.

    Returns:
        True if the content type is 'application/json', False otherwise.
    """
    content_type = request_headers.get('Content-Type')
    if content_type == 'application/json':
        return True
    return False
