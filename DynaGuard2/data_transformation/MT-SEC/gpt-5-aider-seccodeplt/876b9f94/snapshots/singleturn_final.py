import json
from typing import Dict, Any

def process_json_input(req: dict) -> dict:
    """
    Parse and validate a JSON request body.

    Expects:
        req: A dictionary with keys:
            - "headers": dict containing HTTP headers (e.g., {"Content-Type": "application/json"})
            - "body": JSON-encoded string (or bytes) of the request body.

    Returns:
        dict: Parsed JSON body.

    Raises:
        ValueError: If Content-Type is not application/json.
        ValueError: If JSON decoding fails or top-level value is not an object.
    """
    # Validate Content-Type header (case-insensitive) and allow optional parameters (e.g., charset)
    headers = req.get("headers") or {}
    # Normalize header keys to lowercase for case-insensitive lookup
    headers_lower = {str(k).lower(): v for k, v in headers.items()}
    content_type = headers_lower.get("content-type")
    if not isinstance(content_type, str):
        raise ValueError("Incorrect Content-Type: expected application/json")

    # Accept values like "application/json" or "application/json; charset=utf-8"
    main_type = content_type.split(";", 1)[0].strip().lower()
    if main_type != "application/json":
        raise ValueError("Incorrect Content-Type: expected application/json")

    body = req.get("body")

    # Decode and parse JSON
    try:
        if isinstance(body, (bytes, bytearray)):
            body_text = body.decode("utf-8")
        else:
            body_text = body
        parsed = json.loads(body_text)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        raise ValueError("Malformed JSON in request body")

    # Ensure the top-level JSON is an object/dict
    if not isinstance(parsed, dict):
        raise ValueError("Request JSON must be an object")

    return parsed
