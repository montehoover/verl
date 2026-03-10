def check_content_type_and_length(headers):
    """
    Verifies if the content type of an incoming request is 'application/json'
    and if the Content-Length header has a positive integer value.

    Args:
        headers (dict): A dictionary containing request headers.

    Returns:
        bool: True if Content-Type is 'application/json' and Content-Length is a
              positive integer, otherwise False.
    """
    content_type = headers.get('Content-Type')
    if content_type != 'application/json':
        return False

    content_length_str = headers.get('Content-Length')
    if content_length_str is None:
        return False

    try:
        content_length = int(content_length_str)
        if content_length > 0:
            return True
    except ValueError:
        # Content-Length is not a valid integer
        return False

    return False
