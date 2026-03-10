def check_content_type_and_length(headers):
    """
    Verify that:
      - Content-Type is 'application/json'
      - Content-Length is a positive integer

    Args:
        headers (dict): Request headers.

    Returns:
        bool: True only if both conditions are met, else False.
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
    is_json = False
    for part in content_type_value.split(','):
        media_type = part.split(';', 1)[0].strip().lower()
        if media_type == 'application/json':
            is_json = True
            break

    if not is_json:
        return False

    # Find the Content-Length header in a case-insensitive manner
    content_length_value = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == 'content-length':
            content_length_value = value
            break

    if content_length_value is None:
        return False

    # Handle potential list/tuple values for Content-Length
    if isinstance(content_length_value, (list, tuple)):
        candidate = None
        for v in content_length_value:
            if isinstance(v, int):
                candidate = v
                break
            if isinstance(v, str) and v.strip():
                candidate = v
                break
        content_length_value = candidate
        if content_length_value is None:
            return False

    # Parse and validate Content-Length
    if isinstance(content_length_value, int):
        return content_length_value > 0

    if isinstance(content_length_value, str):
        s = content_length_value.strip()
        if s.startswith('"') and s.endswith('"') and len(s) >= 2:
            s = s[1:-1].strip()
        try:
            length = int(s)
        except (ValueError, TypeError):
            return False
        return length > 0

    return False
