def check_content_type(headers):
    """
    Verify if the Content-Type of the request is 'application/json'.

    Args:
        headers (dict): Request headers.

    Returns:
        bool: True if Content-Type is 'application/json', else False.
    """
    if not isinstance(headers, dict):
        return False

    # Find the Content-Type header in a case-insensitive manner
    content_type_value = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == 'content-type':
            content_type_value = value
            break

    if content_type_value is None:
        return False

    # Handle potential list/tuple values
    if isinstance(content_type_value, (list, tuple)):
        content_type_value = next((v for v in content_type_value if isinstance(v, str) and v.strip()), None)
        if content_type_value is None:
            return False

    if not isinstance(content_type_value, str):
        return False

    # Consider parameters and possible comma-separated values
    for part in content_type_value.split(','):
        media_type = part.split(';', 1)[0].strip().lower()
        if media_type == 'application/json':
            return True

    return False
