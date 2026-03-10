def check_content_type_and_length(headers: dict) -> bool:
    """
    Verify that:
      - The Content-Type header is application/json (case-insensitive), ignoring any parameters like charset.
      - The Content-Length header is a positive integer (> 0).
    Returns True only if both conditions are met; otherwise False.
    """
    if not isinstance(headers, dict):
        return False

    content_type_value = None
    content_length_value = None

    # Find headers (case-insensitive)
    for key, value in headers.items():
        if not isinstance(key, str):
            continue
        k = key.lower()
        if k == 'content-type' and content_type_value is None:
            content_type_value = value
        elif k == 'content-length' and content_length_value is None:
            content_length_value = value

    # Validate Content-Type
    if content_type_value is None:
        return False

    if isinstance(content_type_value, (list, tuple)):
        if not content_type_value:
            return False
        content_type_value = content_type_value[0]

    if not isinstance(content_type_value, str):
        return False

    media_type = content_type_value.split(';', 1)[0].strip().lower()
    if media_type != 'application/json':
        return False

    # Validate Content-Length
    if content_length_value is None:
        return False

    if isinstance(content_length_value, (list, tuple)):
        if not content_length_value:
            return False
        content_length_value = content_length_value[0]

    # Convert to int safely
    try:
        if isinstance(content_length_value, (bytes, bytearray)):
            content_length_value = content_length_value.decode('ascii', 'strict')

        if isinstance(content_length_value, str):
            # If multiple values provided, take the first
            content_length_value = content_length_value.split(',', 1)[0].strip()
            if content_length_value == '':
                return False
            length = int(content_length_value)
        elif isinstance(content_length_value, int):
            length = content_length_value
        else:
            return False
    except (ValueError, TypeError, UnicodeDecodeError):
        return False

    return length > 0
