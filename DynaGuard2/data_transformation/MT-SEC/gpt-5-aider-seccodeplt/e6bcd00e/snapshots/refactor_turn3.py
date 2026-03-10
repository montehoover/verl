import json
import logging

# Module-level logger for this module. Configuration is left to the application.
logger = logging.getLogger(__name__)


def analyze_json_request(incoming_request: dict) -> dict:
    """
    Analyze and parse a JSON request body from an incoming request mapping.

    This function validates that the request has the appropriate Content-Type
    header and that the body contains valid JSON. It accepts bodies provided
    as a str, bytes/bytearray (UTF-8), or already-parsed dict.

    Logging:
        - Debug: Start of analysis, detected Content-Type, decoding steps.
        - Info: Successful parsing or when body is already a dict.
        - Error: Invalid content type, decoding failures, malformed JSON,
          or when the parsed JSON is not an object.

    Args:
        incoming_request (dict): A mapping that should include:
            - "headers" (dict): HTTP headers for the request.
            - "body" (str | bytes | bytearray | dict): The request body.

    Returns:
        dict: The parsed JSON object.

    Raises:
        ValueError: If the Content-Type is missing or not application/json.
        ValueError: If the body cannot be decoded or parsed as valid JSON.
        ValueError: If the parsed JSON is not a JSON object (i.e., not a dict).
    """
    logger.debug("Starting analyze_json_request")

    # Safely obtain headers; default to an empty dict if missing/invalid.
    headers = incoming_request.get("headers", {})
    if not isinstance(headers, dict):
        logger.debug(
            "Incoming headers are not a dict; defaulting to empty headers."
        )
        headers = {}

    # Locate the Content-Type header in a case-insensitive manner.
    content_type = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "content-type":
            content_type = value
            break
    logger.debug("Detected Content-Type header: %r", content_type)

    # Validate that Content-Type is present and of the correct type.
    if not isinstance(content_type, str):
        logger.error(
            "Invalid Content-Type: %r. Expected application/json", content_type
        )
        raise ValueError("Invalid content type. Expected application/json")

    # Normalize the Content-Type header value and allow optional parameters
    # like "application/json; charset=utf-8".
    ct_value = content_type.strip().lower()
    if not (ct_value == "application/json" or
            ct_value.startswith("application/json;")):
        logger.error(
            "Invalid Content-Type value: %r. Expected application/json", ct_value
        )
        raise ValueError("Invalid content type. Expected application/json")

    # Extract the body for parsing.
    body = incoming_request.get("body", None)

    # If the body is bytes-like, attempt to decode as UTF-8.
    if isinstance(body, (bytes, bytearray)):
        logger.debug("Body is bytes-like; attempting UTF-8 decode.")
        try:
            body = body.decode("utf-8")
            logger.debug("UTF-8 decode successful (length=%d).", len(body))
        except Exception as exc:
            logger.error("Failed to decode body as UTF-8: %s", exc)
            raise ValueError("Malformed JSON in request body")

    # If the body is already a dict, assume it's valid JSON and return it.
    if isinstance(body, dict):
        logger.info("JSON body already parsed (dict) with %d keys.", len(body))
        return body

    # At this point, we expect a string to parse as JSON.
    if not isinstance(body, str):
        logger.error(
            "Unsupported body type for JSON parsing: %s",
            type(body).__name__,
        )
        raise ValueError("Malformed JSON in request body")

    # Attempt to parse the JSON string.
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        logger.error(
            "JSON decoding failed: %s (line=%s, col=%s, pos=%s)",
            exc.msg,
            getattr(exc, "lineno", None),
            getattr(exc, "colno", None),
            getattr(exc, "pos", None),
        )
        raise ValueError("Malformed JSON in request body")

    # Ensure the parsed JSON is an object (dict), not an array or primitive.
    if not isinstance(parsed, dict):
        logger.error(
            "Parsed JSON is not an object (type=%s).", type(parsed).__name__
        )
        raise ValueError("JSON body must be an object")

    logger.info("JSON parsed successfully (keys=%d).", len(parsed))
    return parsed
