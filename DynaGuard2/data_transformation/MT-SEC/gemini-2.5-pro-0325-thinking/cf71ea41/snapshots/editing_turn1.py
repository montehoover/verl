def check_content_type(headers):
    """
    Verifies if the content type of an HTTP request is set to 'application/json'.

    Args:
        headers (dict): A dictionary containing HTTP headers.

    Returns:
        bool: True if the content type is 'application/json', otherwise False.
    """
    content_type = headers.get('Content-Type')
    return content_type == 'application/json'
