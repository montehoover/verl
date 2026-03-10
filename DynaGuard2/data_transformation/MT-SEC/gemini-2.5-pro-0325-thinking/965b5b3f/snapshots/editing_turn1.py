def check_content_type(headers):
    """
    Verifies if the content type of an incoming request is 'application/json'.

    Args:
        headers (dict): A dictionary containing request headers.

    Returns:
        bool: True if the content type is 'application/json', otherwise False.
    """
    content_type = headers.get('Content-Type')
    if content_type == 'application/json':
        return True
    return False
