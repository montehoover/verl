import json


def process_json_request(req_data: dict) -> dict:
    """
    Parse and validate an incoming JSON request.

    Args:
        req_data (dict): A dictionary containing request data. Expected keys:
            - "headers": dict with request headers (case-insensitive keys supported).
            - "body": bytes | str | dict representing the request body.

    Returns:
        dict: The parsed and validated JSON body.

    Raises:
        ValueError: If the content type is not acceptable.
        ValueError: If the JSON body is invalid or cannot be parsed into a dictionary.
    """
    if not isinstance(req_data, dict):
        raise ValueError("Invalid request data.")

    headers = req_data.get("headers") or {}
    if not isinstance(headers, dict):
        headers = {}

    # Normalize headers to be case-insensitive
    headers_lc = {str(k).lower(): v for k, v in headers.items()}
    content_type_raw = str(headers_lc.get("content-type", "")).strip()

    # Extract MIME type and parameters, e.g., "application/json; charset=utf-8"
    mime_type = ""
    charset = "utf-8"
    if content_type_raw:
        parts = [p.strip() for p in content_type_raw.split(";")]
        mime_type = parts[0].lower()
        for p in parts[1:]:
            if p.lower().startswith("charset="):
                charset = p.split("=", 1)[1].strip() or charset

    # Validate acceptable content types: application/json and application/*+json
    acceptable = mime_type == "application/json" or (
        mime_type.startswith("application/") and mime_type.endswith("+json")
    )
    if not acceptable:
        raise ValueError("Request content type is not acceptable.")

    # Obtain raw body
    body = req_data.get("body", None)

    # Normalize body to string JSON text
    if isinstance(body, dict):
        # Already a parsed dict; return as-is
        return body

    if body is None:
        raise ValueError("Invalid JSON format: request body is empty.")

    if isinstance(body, bytes):
        try:
            body_text = body.decode(charset, errors="strict")
        except (LookupError, UnicodeDecodeError):
            # Unknown charset or decode error -> treat as invalid JSON format
            raise ValueError("Invalid JSON format: unable to decode request body.")
    elif isinstance(body, str):
        body_text = body
    else:
        # Unsupported body type
        raise ValueError("Invalid JSON format: unsupported body type.")

    body_text = body_text.strip()
    if not body_text:
        raise ValueError("Invalid JSON format: request body is empty.")

    # Parse JSON
    try:
        parsed = json.loads(body_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e.msg}") from None

    if not isinstance(parsed, dict):
        raise ValueError("Invalid JSON format: top-level JSON value must be an object.")

    return parsed
