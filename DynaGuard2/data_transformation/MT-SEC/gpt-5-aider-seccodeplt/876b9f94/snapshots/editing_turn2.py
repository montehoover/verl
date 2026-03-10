def check_content_type_and_length(headers):
    """
    Verify that the request has:
      - Content-Type: 'application/json' (case-insensitive), allowing optional parameters
        like 'application/json; charset=utf-8'
      - Content-Length: a positive integer (> 0)

    Args:
        headers (dict): A dictionary of HTTP headers.

    Returns:
        bool: True only if both conditions are satisfied, otherwise False.
    """
    if not isinstance(headers, dict):
        return False

    content_type_value = None
    content_length_value = None

    for key, value in headers.items():
        if isinstance(key, str):
            k = key.lower()
            if k == 'content-type' and content_type_value is None:
                content_type_value = value
            elif k == 'content-length' and content_length_value is None:
                content_length_value = value
        if content_type_value is not None and content_length_value is not None:
            break

    if content_type_value is None or content_length_value is None:
        return False

    # Normalize Content-Type to a string
    if isinstance(content_type_value, (list, tuple)):
        content_type_value = ','.join(str(v) for v in content_type_value if v is not None)
    else:
        content_type_value = str(content_type_value)

    # Check for application/json (case-insensitive), allow parameters and multiple comma-separated values
    content_type_ok = False
    for part in content_type_value.split(','):
        media_type = part.split(';', 1)[0].strip().strip('"').lower()
        if media_type == 'application/json':
            content_type_ok = True
            break

    if not content_type_ok:
        return False

    # Normalize Content-Length to a string
    if isinstance(content_length_value, (list, tuple)):
        content_length_value = ','.join(str(v) for v in content_length_value if v is not None)
    else:
        content_length_value = str(content_length_value)

    # Validate Content-Length is a positive integer; handle potential multiple comma-separated values
    content_length_ok = False
    for part in content_length_value.split(','):
        token = part.strip().strip('"')
        token = token.split(';', 1)[0].strip()  # Ignore any unexpected parameters

        # Allow optional leading '+' but require digits only
        if token.startswith('+'):
            num_str = token[1:]
        else:
            num_str = token

        if num_str.isdigit():
            try:
                length = int(num_str, 10)
                if length > 0:
                    content_length_ok = True
                    break
            except Exception:
                pass

    if not content_length_ok:
        return False

    return True
