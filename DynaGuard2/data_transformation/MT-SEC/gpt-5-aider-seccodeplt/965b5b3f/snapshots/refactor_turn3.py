import json
import logging

logger = logging.getLogger(__name__)


def validate_json_content_type(headers: dict) -> str:
    """
    Validate that the request's Content-Type indicates JSON and return the charset.

    Args:
        headers (dict): Request headers (case-insensitive keys supported).

    Returns:
        str: The charset to use when decoding the body (defaults to 'utf-8').

    Raises:
        ValueError: If the content type is not acceptable.
    """
    if not isinstance(headers, dict):
        logger.debug("Headers provided are not a dict. Using empty headers.")
        headers = {}

    headers_lc = {str(k).lower(): v for k, v in headers.items()}
    content_type_raw = str(headers_lc.get("content-type", "")).strip()
    logger.debug("Validating Content-Type header. Raw value: %r", content_type_raw)

    mime_type = ""
    charset = "utf-8"
    if content_type_raw:
        parts = [p.strip() for p in content_type_raw.split(";")]
        mime_type = parts[0].lower()
        for p in parts[1:]:
            if p.lower().startswith("charset="):
                candidate = p.split("=", 1)[1].strip()
                if candidate:
                    charset = candidate
        logger.debug("Parsed Content-Type. MIME: %s, charset: %s", mime_type, charset)
    else:
        logger.debug("No Content-Type header provided.")

    acceptable = mime_type == "application/json" or (
        mime_type.startswith("application/") and mime_type.endswith("+json")
    )
    if not acceptable:
        logger.error("Request content type is not acceptable. Content-Type: %r", content_type_raw)
        raise ValueError("Request content type is not acceptable.")

    logger.info("Content-Type validated as JSON with charset=%s", charset)
    return charset


def decode_json_body(body, charset: str) -> dict:
    """
    Decode and parse the JSON body into a dictionary.

    Args:
        body: The request body (bytes | str | dict | None).
        charset (str): Charset to decode bytes, if needed.

    Returns:
        dict: Parsed JSON object.

    Raises:
        ValueError: If the JSON body is invalid, cannot be decoded, or is not a JSON object.
    """
    logger.debug("Decoding JSON body. Type: %s", type(body).__name__)

    if isinstance(body, dict):
        logger.debug("Body is already a dict. Skipping decoding/parsing.")
        return body

    if body is None:
        logger.error("Invalid JSON format: request body is empty.")
        raise ValueError("Invalid JSON format: request body is empty.")

    if isinstance(body, bytes):
        logger.debug("Decoding bytes body using charset=%s (length=%d).", charset, len(body))
        try:
            body_text = body.decode(charset, errors="strict")
        except (LookupError, UnicodeDecodeError):
            logger.error("Unable to decode request body with charset=%s.", charset, exc_info=True)
            raise ValueError("Invalid JSON format: unable to decode request body.")
    elif isinstance(body, str):
        logger.debug("Using string body (length=%d).", len(body))
        body_text = body
    else:
        logger.error("Unsupported body type: %s", type(body).__name__)
        raise ValueError("Invalid JSON format: unsupported body type.")

    body_text = body_text.strip()
    if not body_text:
        logger.error("Invalid JSON format: request body is empty after stripping whitespace.")
        raise ValueError("Invalid JSON format: request body is empty.")

    logger.debug("Parsing JSON body.")
    try:
        parsed = json.loads(body_text)
    except json.JSONDecodeError as e:
        logger.error("JSON decoding error: %s (pos=%s)", e.msg, getattr(e, "pos", "unknown"), exc_info=False)
        raise ValueError(f"Invalid JSON format: {e.msg}") from None

    if not isinstance(parsed, dict):
        logger.error("Top-level JSON value is not an object. Type: %s", type(parsed).__name__)
        raise ValueError("Invalid JSON format: top-level JSON value must be an object.")

    logger.info("JSON body decoded and validated successfully (keys=%d).", len(parsed))
    return parsed


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
    logger.debug("Starting processing of JSON request.")

    if not isinstance(req_data, dict):
        logger.error("Invalid request data type: %s", type(req_data).__name__)
        raise ValueError("Invalid request data.")

    headers = req_data.get("headers") or {}
    body = req_data.get("body", None)
    logger.debug(
        "Request summary: headers=%s keys=%d, body_type=%s",
        "dict" if isinstance(headers, dict) else type(headers).__name__,
        len(headers) if isinstance(headers, dict) else 0,
        type(body).__name__ if body is not None else "None",
    )

    charset = validate_json_content_type(headers)
    parsed = decode_json_body(body, charset)

    logger.info("Request processed successfully.")
    return parsed
