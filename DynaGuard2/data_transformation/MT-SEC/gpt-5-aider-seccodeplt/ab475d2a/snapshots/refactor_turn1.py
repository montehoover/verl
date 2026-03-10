import json
from typing import Any, Dict


def decode_json_request(req: dict) -> dict:
    """
    Parse and validate the body of an incoming JSON request.

    Args:
        req: dict containing at least 'headers' (dict) and 'body' (bytes or str).

    Returns:
        dict: Parsed JSON object.

    Raises:
        ValueError: If the content type is not acceptable, the body cannot be decoded,
                    the JSON is invalid, or the decoded JSON is not an object.
    """
    if not isinstance(req, dict):
        raise ValueError("Invalid request object.")

    headers = req.get("headers", {})
    if not isinstance(headers, dict):
        raise ValueError("Invalid request headers.")

    # Normalize header keys to lowercase for case-insensitive access
    headers_lc: Dict[str, Any] = {str(k).lower(): v for k, v in headers.items()}

    # Obtain content-type; support common variations
    content_type = headers_lc.get("content-type") or headers_lc.get("content_type")
    if not content_type or not isinstance(content_type, str):
        raise ValueError("Unsupported content type; expected JSON.")

    # Extract media type and optional parameters (e.g., charset)
    ct = content_type.strip().lower()
    # Some servers might return multiple values; take the first
    ct = ct.split(",")[0].strip()
    media_type, *params = [p.strip() for p in ct.split(";")]
    acceptable = media_type == "application/json" or media_type.endswith("+json")
    if not acceptable:
        raise ValueError("Unsupported content type; expected JSON.")

    # Default charset to UTF-8 unless otherwise specified
    charset = "utf-8"
    for p in params:
        if p.startswith("charset="):
            charset = p.split("=", 1)[1].strip() or "utf-8"
            break

    body = req.get("body", b"")
    # Normalize body to text
    if isinstance(body, bytes):
        try:
            text = body.decode(charset)
        except (LookupError, UnicodeDecodeError):
            raise ValueError("Unable to decode request body with declared charset.")
    elif isinstance(body, str):
        text = body
    else:
        raise ValueError("Invalid request body type.")

    if not text or not text.strip():
        raise ValueError("Empty JSON body.")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Provide the JSON error message for clarity
        raise ValueError(f"Invalid JSON: {exc.msg}") from None

    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object.")

    return data
