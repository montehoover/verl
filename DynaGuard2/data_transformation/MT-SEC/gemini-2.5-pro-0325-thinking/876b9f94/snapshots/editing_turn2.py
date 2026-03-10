def check_content_type_and_length(headers: dict) -> bool:
    """
    Verifies if the Content-Type is 'application/json' and Content-Length is a positive integer.

    Args:
        headers: A dictionary containing HTTP headers.

    Returns:
        True if Content-Type is 'application/json' and Content-Length is a positive integer,
        otherwise False.
    """
    content_type = headers.get('Content-Type')
    if content_type != 'application/json':
        return False

    content_length_str = headers.get('Content-Length')
    if content_length_str is None:
        return False

    try:
        content_length = int(content_length_str)
    except ValueError:
        return False

    if content_length > 0:
        return True
    
    return False
