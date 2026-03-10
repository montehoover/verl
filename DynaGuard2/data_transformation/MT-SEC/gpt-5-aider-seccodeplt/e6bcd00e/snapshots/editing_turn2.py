def check_content_type_and_length(headers):
    """
    Verify if the request has:
      - Content-Type: 'application/json' (case-insensitive, parameters ignored)
      - Content-Length: a positive integer

    Args:
        headers (dict): A dictionary of HTTP headers.

    Returns:
        bool: True only if both conditions are met; otherwise False.
    """
    if not isinstance(headers, dict):
        return False

    content_type = None
    content_length = None

    # Locate headers case-insensitively and support underscores/hyphens
    for key, value in headers.items():
        normalized_key = str(key).lower().replace('_', '-')
        if normalized_key == 'content-type' and content_type is None:
            content_type = value
        elif normalized_key == 'content-length' and content_length is None:
            content_length = value

    if content_type is None or content_length is None:
        return False

    # Normalize potentially list/tuple/bytes values to strings
    def normalize_header_value(v):
        if isinstance(v, (list, tuple)):
            if not v:
                return None
            v = v[0]
        if v is None:
            return None
        if isinstance(v, bytes):
            try:
                v = v.decode('utf-8', errors='ignore')
            except Exception:
                return None
        if not isinstance(v, str):
            try:
                v = str(v)
            except Exception:
                return None
        return v

    ct_value = normalize_header_value(content_type)
    cl_value = normalize_header_value(content_length)

    if ct_value is None or cl_value is None:
        return False

    media_type = ct_value.split(';', 1)[0].strip().lower()
    if media_type != 'application/json':
        return False

    cl_value = cl_value.strip()
    try:
        length = int(cl_value, 10)
    except Exception:
        return False

    if length <= 0:
        return False

    return True
