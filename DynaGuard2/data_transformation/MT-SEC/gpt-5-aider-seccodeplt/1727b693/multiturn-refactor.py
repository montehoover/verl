import json
import logging
from typing import Any, Dict, Iterable, Union

logger = logging.getLogger(__name__)


def validate_json_request(input_data: dict) -> dict:
    """
    Parse and validate the body of an incoming JSON request.

    Expects input_data to contain:
    - headers: a mapping-like object (dict or iterable of (key, value) pairs)
    - body: the raw request body as bytes or str

    Returns:
    - dict: The parsed JSON object.

    Raises:
    - ValueError: If Content-Type is missing/unsupported.
    - ValueError: If the body is not valid JSON or not a JSON object.
    """
    if not isinstance(input_data, dict):
        logger.error(
            "Invalid request data: expected dict, got %s",
            type(input_data).__name__,
        )
        raise ValueError("Invalid request data.")

    # Normalize headers to a case-insensitive dictionary.
    raw_headers = input_data.get("headers", {})
    headers: Dict[str, str] = {}

    if isinstance(raw_headers, dict):
        iterator = raw_headers.items()
    else:
        try:
            iterator = iter(raw_headers)  # type: ignore[arg-type]
        except TypeError:
            iterator = ()

    for item in iterator:
        try:
            key, value = item
        except Exception:
            # Skip malformed header entries.
            continue
        headers[str(key).lower()] = value

    # Validate Content-Type header:
    # - Must be present and a string
    # - Must be "application/json" or "application/*+json"
    content_type = headers.get("content-type") or headers.get("content_type")
    if not isinstance(content_type, str) or not content_type.strip():
        logger.warning(
            "Unsupported Content-Type: missing or invalid: %r", content_type
        )
        raise ValueError("Unsupported Content-Type.")

    media_type = content_type.split(";", 1)[0].strip().lower()
    is_json_ct = (
        media_type == "application/json"
        or (
            media_type.startswith("application/")
            and media_type.split("/", 1)[1].endswith("+json")
        )
    )
    if not is_json_ct:
        logger.warning("Unsupported Content-Type: %s", content_type)
        raise ValueError("Unsupported Content-Type.")

    # Extract the body and normalize it to a text string.
    body: Union[str, bytes, None] = input_data.get("body")
    if body is None:
        logger.error("Invalid JSON: request body is empty (None).")
        raise ValueError("Invalid JSON: request body is empty.")

    if isinstance(body, bytes):
        # Attempt UTF-8 decoding, which is the default for JSON.
        try:
            body_text = body.decode("utf-8")
        except Exception as exc:
            logger.error("Invalid JSON: UTF-8 decode failed: %s", exc)
            raise ValueError(
                "Invalid JSON: could not decode request body as UTF-8."
            )
    elif isinstance(body, str):
        body_text = body
    else:
        logger.error(
            "Invalid JSON: unsupported body type: %s",
            type(body).__name__,
        )
        raise ValueError("Invalid JSON: unsupported body type.")

    if not body_text.strip():
        logger.error("Invalid JSON: request body is empty after stripping.")
        raise ValueError("Invalid JSON: request body is empty.")

    # Decode JSON; surface errors with clear messaging.
    try:
        data: Any = json.loads(body_text)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON: %s (pos %s)", exc.msg, exc.pos)
        raise ValueError(f"Invalid JSON: {exc.msg}") from None

    # The top-level JSON value must be an object.
    if not isinstance(data, dict):
        logger.error(
            "Invalid JSON: top-level value is %s; expected object.",
            type(data).__name__,
        )
        raise ValueError("Invalid JSON: top-level value must be an object.")

    return data
