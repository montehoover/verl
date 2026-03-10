import json

def process_json_input(req):
    """
    Parse a JSON request body from a request-like dict.

    Args:
        req (dict): Contains 'headers' (dict-like) and 'body' (bytes or str).

    Returns:
        dict: Parsed JSON object.

    Raises:
        ValueError: If Content-Type is not application/json or if JSON is malformed.
    """
    headers = req.get('headers', {})
    if not isinstance(headers, dict):
        headers = {}

    # Extract Content-Type (case-insensitive)
    content_type_value = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == 'content-type':
            content_type_value = value
            break

    if content_type_value is None:
        raise ValueError("Invalid Content-Type: expected 'application/json'.")

    # Normalize Content-Type to a string
    if isinstance(content_type_value, (list, tuple)):
        content_type_str = ','.join(str(v) for v in content_type_value if v is not None)
    else:
        content_type_str = str(content_type_value)

    # Check for application/json (allow parameters and multiple comma-separated values)
    is_json = False
    for part in content_type_str.split(','):
        media_type = part.split(';', 1)[0].strip().strip('"').lower()
        if media_type == 'application/json':
            is_json = True
            break

    if not is_json:
        raise ValueError("Invalid Content-Type: expected 'application/json'.")

    # Obtain and decode the body
    raw_body = req.get('body', b'')
    if isinstance(raw_body, (bytes, bytearray)):
        try:
            text = raw_body.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError("Malformed JSON: could not decode request body.")
    elif isinstance(raw_body, str):
        text = raw_body
    else:
        text = str(raw_body)

    # Parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("Malformed JSON: could not decode request body.")

    if not isinstance(data, dict):
        raise ValueError("Malformed JSON: expected a JSON object.")

    return data
