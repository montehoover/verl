def check_content_type(headers):
    """
    Verify whether the Content-Type header is 'application/json' (case-insensitive),
    allowing optional parameters (e.g., 'application/json; charset=utf-8').

    Args:
        headers (dict): A dictionary of HTTP headers.

    Returns:
        bool: True if Content-Type is 'application/json', otherwise False.
    """
    if not isinstance(headers, dict):
        return False

    content_type_value = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == 'content-type':
            content_type_value = value
            break

    if content_type_value is None:
        return False

    # Normalize to a string
    if isinstance(content_type_value, (list, tuple)):
        content_type_value = ','.join(str(v) for v in content_type_value if v is not None)
    else:
        content_type_value = str(content_type_value)

    # Handle multiple values separated by commas
    for part in content_type_value.split(','):
        media_type = part.split(';', 1)[0].strip().strip('"').lower()
        if media_type == 'application/json':
            return True

    return False
