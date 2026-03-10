def check_content_type(request_headers):
    """
    Verify if the Content-Type header is 'application/json'.

    Args:
        request_headers (dict): A dictionary of HTTP headers.

    Returns:
        bool: True if Content-Type is 'application/json', otherwise False.
    """
    if not isinstance(request_headers, dict):
        return False

    content_type = None
    for key, value in request_headers.items():
        normalized_key = key.lower().replace('_', '-')
        if normalized_key == 'content-type':
            content_type = value
            break

    if content_type is None:
        return False

    # Some frameworks may provide header values as lists/tuples
    if isinstance(content_type, (list, tuple)):
        if not content_type:
            return False
        content_type = content_type[0]

    if not isinstance(content_type, str):
        try:
            content_type = str(content_type)
        except Exception:
            return False

    media_type = content_type.split(';', 1)[0].strip().lower()
    return media_type == 'application/json'
