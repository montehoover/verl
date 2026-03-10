import json


def process_json_input(req: dict) -> dict:
    """
    Parse and validate a JSON request body from a FastAPI-style request dictionary.

    Expects:
      - req to be a dict containing:
          - 'headers': dict of HTTP headers (case-insensitive keys)
          - 'body': JSON payload as str/bytes or already-parsed dict

    Returns:
      - dict: Parsed JSON object.

    Raises:
      - ValueError: If Content-Type is missing/incorrect, body is malformed JSON,
                    or top-level JSON is not an object.
    """
    if not isinstance(req, dict):
        raise ValueError("Request must be a dictionary containing 'headers' and 'body'")

    headers = req.get('headers', {})
    if not isinstance(headers, dict):
        raise ValueError("Request headers must be a dictionary")

    # Locate Content-Type header (case-insensitive)
    content_type = None
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == 'content-type':
            content_type = v
            break

    if not content_type:
        raise ValueError("Missing Content-Type header; expected 'application/json'")
    if not isinstance(content_type, str):
        raise ValueError("Invalid Content-Type header value; expected a string")

    # Parse media type and optional parameters (e.g., charset)
    parts = [p.strip() for p in content_type.split(';') if p.strip()]
    media_type = parts[0].lower() if parts else ''
    if media_type != 'application/json':
        raise ValueError(f"Invalid Content-Type: expected 'application/json', got '{content_type}'")

    charset = 'utf-8'
    for p in parts[1:]:
        if '=' in p:
            name, val = p.split('=', 1)
            if name.strip().lower() == 'charset':
                charset = (val.strip().strip('"').strip("'")) or 'utf-8'
                break

    body = req.get('body', None)

    # If body is already a dict, accept it as parsed JSON
    if isinstance(body, dict):
        return body

    # Convert body to string as needed
    if isinstance(body, (bytes, bytearray, memoryview)):
        try:
            body_str = bytes(body).decode(charset)
        except (LookupError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid character encoding for request body: {e}") from e
    elif isinstance(body, str):
        body_str = body
    elif body is None:
        raise ValueError("Malformed JSON in request body")
    else:
        raise ValueError("Request body must be a JSON string or bytes")

    # Parse JSON
    try:
        data = json.loads(body_str)
    except json.JSONDecodeError as e:
        raise ValueError("Malformed JSON in request body") from e

    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")

    return data
