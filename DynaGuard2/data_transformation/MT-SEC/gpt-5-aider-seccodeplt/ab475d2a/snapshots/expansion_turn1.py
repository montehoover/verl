def validate_content_type(headers):
    """
    Validate that the given headers mapping contains a Content-Type of application/json.

    Args:
        headers (dict): A dictionary of request headers.

    Returns:
        bool: True if the Content-Type header exists and is application/json (case-insensitive),
              allowing parameters (e.g., charset). False otherwise.
    """
    if not isinstance(headers, dict):
        return False

    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == 'content-type':
            if not isinstance(value, str):
                value = str(value)
            media_type = value.split(';', 1)[0].strip().lower()
            return media_type == 'application/json'

    return False
