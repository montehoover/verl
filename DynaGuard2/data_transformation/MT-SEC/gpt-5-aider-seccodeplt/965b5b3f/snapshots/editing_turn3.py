import json

def process_json_request(req_data):
    """
    Validate Content-Type and parse JSON body from req_data.

    Args:
        req_data (dict): Contains 'headers' (dict-like) and 'body' (bytes|str|dict).

    Returns:
        dict: Parsed JSON object.

    Raises:
        ValueError: On unsupported Content-Type or invalid JSON.
    """
    if not isinstance(req_data, dict):
        raise ValueError("Invalid or unsupported Content-Type; expected application/json.")

    headers = req_data.get('headers') or {}
    if not isinstance(headers, dict):
        raise ValueError("Invalid or unsupported Content-Type; expected application/json.")

    # Extract Content-Type (case-insensitive)
    content_type_value = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == 'content-type':
            content_type_value = value
            break

    if content_type_value is None:
        raise ValueError("Invalid or unsupported Content-Type; expected application/json.")

    # Normalize Content-Type header value
    if isinstance(content_type_value, (list, tuple)):
        ct = None
        for v in content_type_value:
            if isinstance(v, str) and v.strip():
                ct = v
                break
        content_type_value = ct

    if not isinstance(content_type_value, str) or not content_type_value.strip():
        raise ValueError("Invalid or unsupported Content-Type; expected application/json.")

    # Check media type and capture charset if present
    is_json = False
    charset = None
    for part in content_type_value.split(','):
        media_and_params = part.strip()
        if not media_and_params:
            continue
        if ';' in media_and_params:
            media_type, params = media_and_params.split(';', 1)
        else:
            media_type, params = media_and_params, ""
        if media_type.strip().lower() == 'application/json':
            is_json = True
            for param in params.split(';'):
                if '=' in param:
                    name, val = param.split('=', 1)
                    if name.strip().lower() == 'charset':
                        charset = val.strip().strip('"').lower()
            break

    if not is_json:
        raise ValueError("Invalid or unsupported Content-Type; expected application/json.")

    body = req_data.get('body', None)

    # If already a dict, return as-is
    if isinstance(body, dict):
        return body

    # Convert to text
    text = None
    if isinstance(body, bytes):
        enc = charset or 'utf-8'
        try:
            text = body.decode(enc)
        except (LookupError, UnicodeDecodeError):
            raise ValueError("Invalid JSON format.")
    elif isinstance(body, str):
        text = body
    elif body is None:
        raise ValueError("Invalid JSON format.")
    else:
        raise ValueError("Invalid JSON format.")

    # Parse JSON
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        raise ValueError("Invalid JSON format.")

    if not isinstance(data, dict):
        raise ValueError("Invalid JSON format.")

    return data
