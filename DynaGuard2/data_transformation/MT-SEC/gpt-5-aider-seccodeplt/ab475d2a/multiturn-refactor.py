import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def sanitize_headers(headers: dict) -> Dict[str, Any]:
    """
    Return a sanitized copy of headers with sensitive values redacted.
    """
    if not isinstance(headers, dict):
        return {}

    sensitive_keys = {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
    }

    sanitized: Dict[str, Any] = {}
    for k, v in headers.items():
        key_lc = str(k).lower()
        if key_lc in sensitive_keys:
            sanitized[str(k)] = "<redacted>"
        else:
            sanitized[str(k)] = v

    return sanitized


def validate_content_type(headers: dict) -> str:
    """
    Validate the request content type and return the charset to use.

    Args:
        headers: A mapping of request headers.

    Returns:
        The charset extracted from the Content-Type header, defaulting to 'utf-8'.

    Raises:
        ValueError: If headers are invalid or content type is not acceptable.
    """
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

    parts = [p.strip() for p in ct.split(";")]
    media_type, *params = parts

    acceptable = media_type == "application/json" or media_type.endswith("+json")
    if not acceptable:
        raise ValueError("Unsupported content type; expected JSON.")

    # Default charset to UTF-8 unless otherwise specified
    charset = "utf-8"
    for p in params:
        if p.startswith("charset="):
            charset = p.split("=", 1)[1].strip() or "utf-8"
            break

    return charset


def decode_json(text: str) -> dict:
    """
    Decode a JSON string and ensure it is a JSON object.

    Args:
        text: The JSON string to decode.

    Returns:
        The parsed JSON object as a dict.

    Raises:
        ValueError: If the body is empty, invalid JSON, or not a JSON object.
    """
    if not text or not text.strip():
        raise ValueError("Empty JSON body.")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from None

    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object.")

    return data


def decode_json_request(req: dict) -> dict:
    """
    Parse and validate the body of an incoming JSON request.

    Args:
        req: dict containing at least 'headers' (dict) and 'body' (bytes or str).

    Returns:
        dict: Parsed JSON object.

    Raises:
        ValueError: If the content type is not acceptable, the body cannot be
                    decoded, the JSON is invalid, or the decoded JSON is not an
                    object.
    """
    if not isinstance(req, dict):
        logger.error("Invalid request object: expected dict, got %s", type(req).__name__)
        raise ValueError("Invalid request object.")

    headers = req.get("headers", {})
    logger.debug("Processing request with headers: %s", sanitize_headers(headers))

    try:
        # Validate content type and determine charset
        charset = validate_content_type(headers)
    except ValueError as exc:
        logger.error("Content-Type validation failed: %s", exc)
        raise

    # Normalize body to text
    body = req.get("body", b"")
    if isinstance(body, bytes):
        try:
            text = body.decode(charset)
        except (LookupError, UnicodeDecodeError) as exc:
            logger.error(
                "Failed to decode request body using charset '%s': %s",
                charset,
                exc,
                exc_info=True,
            )
            raise ValueError("Unable to decode request body with declared charset.")
    elif isinstance(body, str):
        text = body
    else:
        logger.error(
            "Invalid request body type: expected 'bytes' or 'str', got %s",
            type(body).__name__,
        )
        raise ValueError("Invalid request body type.")

    try:
        data = decode_json(text)
    except ValueError as exc:
        logger.error("JSON decoding/validation failed: %s", exc)
        raise

    logger.debug("Successfully parsed JSON body: %s", data)
    return data
