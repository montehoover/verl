import json

def process_json_payload(req_data):
    """
    Parse and validate an incoming JSON request.

    Requirements:
    - req_data is a dict containing at least:
        - headers: dict of HTTP headers
        - body: request body (bytes, str, or already-parsed dict)
    - Content-Type must be JSON-compatible:
        - application/json
        - application/*+json
        - (optionally) text/json
      Parameters (e.g., charset) are allowed and ignored for media type matching.
    - If body is str/bytes, parse as JSON. If already a dict, return as-is.
    - Return the parsed JSON object (dict).
    - Raise ValueError if Content-Type is not JSON-compatible, body is missing,
      cannot be decoded, is malformed JSON, or is not a JSON object.

    Args:
        req_data (dict): Request data containing headers and body.

    Returns:
        dict: Parsed JSON body.

    Raises:
        ValueError: On invalid/missing headers, content type, body, or malformed JSON.
    """
    if not isinstance(req_data, dict):
        raise ValueError("Invalid request data: expected a dictionary")

    headers = req_data.get("headers")
    if not isinstance(headers, dict):
        raise ValueError("Missing or invalid headers")

    # Normalize header keys (case-insensitive, underscores to hyphens)
    normalized = {}
    for k, v in headers.items():
        key = str(k).replace("_", "-").lower()
        normalized[key] = v

    # Helper to get first non-empty string value from header
    def first_value(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                if item is None:
                    continue
                s = item.decode("utf-8", errors="ignore") if isinstance(item, (bytes, bytearray)) else str(item)
                if s.strip():
                    return s
            return ""
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                return str(value)
        return "" if value is None else str(value)

    ct_raw = normalized.get("content-type")
    if ct_raw is None:
        raise ValueError("Content-Type header is required")

    ct_value = first_value(ct_raw)
    if not ct_value:
        raise ValueError("Content-Type header is required")

    media_type = ct_value.split(";", 1)[0].strip().lower()

    def is_json_mediatype(mt):
        if mt == "application/json":
            return True
        if mt.endswith("+json"):
            return True
        if mt == "text/json":
            return True
        return False

    if not is_json_mediatype(media_type):
        raise ValueError("Unsupported Content-Type: expected application/json or application/*+json")

    # Determine charset if provided
    charset = None
    if ";" in ct_value:
        parts = [p.strip() for p in ct_value.split(";")[1:]]
        for p in parts:
            if "=" in p:
                name, val = p.split("=", 1)
                if name.strip().lower() == "charset":
                    charset = val.strip().strip('"').strip("'")
                    break
    encoding = charset or "utf-8"

    # Extract and parse body
    if "body" not in req_data:
        raise ValueError("Request body is required")

    body = req_data["body"]

    if isinstance(body, dict):
        return body

    if isinstance(body, (bytes, bytearray)):
        try:
            text = body.decode(encoding, errors="strict")
        except Exception:
            raise ValueError(f"Unable to decode request body with charset {encoding}")
    elif isinstance(body, str):
        text = body
    else:
        raise ValueError("Unsupported request body type")

    # Strip BOM if present
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValueError("Malformed JSON payload") from e

    if not isinstance(obj, dict):
        raise ValueError("JSON payload must be a JSON object")

    return obj
