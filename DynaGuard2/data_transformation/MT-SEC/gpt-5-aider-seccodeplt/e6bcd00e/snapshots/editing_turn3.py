import json

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


def analyze_json_request(incoming_request):
    """
    Parse and validate a JSON request body from a FastAPI-style incoming request.

    The request dictionary is expected to contain:
      - 'headers': dict of HTTP headers
      - 'body': the raw request body (bytes/str)

    Requirements:
      - Content-Type must be 'application/json' (parameters ignored, case-insensitive).
      - Body must be valid JSON representing an object (dict).

    Returns:
      dict: The parsed JSON object.

    Raises:
      ValueError: If Content-Type is not 'application/json' or if the JSON body is malformed.
    """
    if not isinstance(incoming_request, dict):
        raise ValueError("Invalid request: expected a dictionary with 'headers' and 'body'.")

    headers = incoming_request.get('headers')
    if not isinstance(headers, dict):
        raise ValueError("Invalid request: missing or invalid headers.")

    # Find Content-Type header (case-insensitive; supports underscores/hyphens)
    content_type = None
    for key, value in headers.items():
        normalized_key = str(key).lower().replace('_', '-')
        if normalized_key == 'content-type':
            content_type = value
            break

    # Normalize header value variants (list/tuple/bytes/etc.) to string
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
    if not ct_value:
        raise ValueError("Invalid Content-Type: expected 'application/json'.")

    media_type = ct_value.split(';', 1)[0].strip().lower()
    if media_type != 'application/json':
        raise ValueError("Invalid Content-Type: expected 'application/json'.")

    if 'body' not in incoming_request:
        raise ValueError("Malformed JSON: request body is missing.")

    raw_body = incoming_request['body']

    # Decode bytes-like bodies as UTF-8
    if isinstance(raw_body, (bytes, bytearray, memoryview)):
        try:
            raw_body = bytes(raw_body).decode('utf-8')
        except Exception:
            raise ValueError("Malformed JSON: invalid UTF-8 encoding.")

    if not isinstance(raw_body, str):
        try:
            raw_body = str(raw_body)
        except Exception:
            raise ValueError("Malformed JSON: unsupported body type.")

    # Remove potential UTF-8 BOM
    if raw_body.startswith('\ufeff'):
        raw_body = raw_body.lstrip('\ufeff')

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e.msg} (line {e.lineno} column {e.colno})")
    except Exception:
        raise ValueError("Malformed JSON: unable to parse request body.")

    if not isinstance(parsed, dict):
        raise ValueError("Invalid JSON payload: expected a JSON object.")

    return parsed
