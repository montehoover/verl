"""
Utilities for parsing and validating JSON request bodies.

This module provides a helper function to parse JSON payloads from a
FastAPI-style request dictionary. It enforces that the Content-Type header
indicates a JSON payload and that the body is valid JSON representing an
object (i.e., a top-level dictionary).

Logging:
    This module uses the standard logging library. It logs:
      - Debug information about detected Content-Type and processing steps.
      - Errors for validation failures and malformed input.
      - Exceptions during byte decoding and JSON parsing with stack traces.
"""

import json
import logging


logger = logging.getLogger(__name__)


def process_json_input(req: dict) -> dict:
    """
    Parse and validate a JSON request body from a FastAPI-style request dict.

    The function expects a dictionary with:
      - headers: A dictionary of HTTP headers (header names are case-insensitive).
      - body: The request payload, which can be:
          - str: JSON string.
          - bytes/bytearray/memoryview: Byte representation of the JSON.
          - dict: Already-parsed JSON object (will be returned as-is).

    The Content-Type header must be 'application/json' (parameters like
    'charset=utf-8' are allowed). If a charset parameter is present, it is used
    to decode bytes; otherwise UTF-8 is assumed.

    Logging:
        - On success, logs an info message with the Content-Type.
        - On failure, logs an error explaining the reason.
        - If byte decoding or JSON parsing fails, logs the exception with
          a stack trace.

    Args:
        req (dict): Request dictionary containing 'headers' and 'body'.

    Returns:
        dict: The parsed JSON object (top-level dictionary).

    Raises:
        ValueError: If:
            - The request or headers are not in the expected structure.
            - The Content-Type header is missing or not 'application/json'.
            - The body is not a string, bytes, or dictionary.
            - The character encoding is invalid or decoding fails.
            - The JSON is malformed.
            - The top-level JSON value is not an object (dictionary).
    """
    # Validate request container.
    if not isinstance(req, dict):
        logger.error(
            "Invalid request container: expected dict, got %s",
            type(req).__name__,
        )
        raise ValueError(
            "Request must be a dictionary containing 'headers' and 'body'"
        )

    # Extract and validate headers.
    headers = req.get("headers", {})
    if not isinstance(headers, dict):
        logger.error(
            "Invalid headers container: expected dict, got %s",
            type(headers).__name__,
        )
        raise ValueError("Request headers must be a dictionary")

    # Locate Content-Type header (case-insensitive).
    content_type = None
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == "content-type":
            content_type = v
            break

    # Ensure Content-Type is present and valid.
    if not content_type:
        logger.error(
            "Missing Content-Type header; expected 'application/json'"
        )
        raise ValueError("Missing Content-Type header; expected 'application/json'")

    if not isinstance(content_type, str):
        logger.error(
            "Invalid Content-Type header value type: %s",
            type(content_type).__name__,
        )
        raise ValueError("Invalid Content-Type header value; expected a string")

    # Parse media type and optional parameters (e.g., charset).
    parts = [p.strip() for p in content_type.split(";") if p.strip()]
    media_type = parts[0].lower() if parts else ""
    logger.debug("Detected Content-Type header: %r (media type: %r)",
                 content_type, media_type)

    if media_type != "application/json":
        logger.error(
            "Invalid Content-Type: expected 'application/json', got %r",
            content_type,
        )
        raise ValueError(
            f"Invalid Content-Type: expected 'application/json', got '{content_type}'"
        )

    # Default to UTF-8 unless an explicit charset is provided.
    charset = "utf-8"
    for p in parts[1:]:
        if "=" in p:
            name, val = p.split("=", 1)
            if name.strip().lower() == "charset":
                charset = (val.strip().strip('"').strip("'")) or "utf-8"
                break

    logger.debug("Using character set for decoding: %s", charset)

    # Retrieve the body.
    body = req.get("body", None)

    # If body is already a dict, accept it as parsed JSON.
    if isinstance(body, dict):
        logger.info(
            "JSON parsing successful (pre-parsed dict). Content-Type: %s",
            content_type,
        )
        return body

    # Convert body to a string if needed.
    if isinstance(body, (bytes, bytearray, memoryview)):
        try:
            body_str = bytes(body).decode(charset)
        except (LookupError, UnicodeDecodeError) as e:
            # LookupError -> unknown charset; UnicodeDecodeError -> decoding failure.
            logger.error(
                "Failed to decode request body with charset %s: %s",
                charset,
                e,
                exc_info=True,
            )
            raise ValueError(
                f"Invalid character encoding for request body: {e}"
            ) from e
    elif isinstance(body, str):
        body_str = body
    elif body is None:
        # Treat missing body as malformed JSON as per contract.
        logger.error("Missing or None request body; cannot parse JSON")
        raise ValueError("Malformed JSON in request body")
    else:
        # Unsupported body type.
        logger.error(
            "Unsupported body type: expected str/bytes, got %s",
            type(body).__name__,
        )
        raise ValueError("Request body must be a JSON string or bytes")

    # Parse JSON into a Python object.
    try:
        data = json.loads(body_str)
    except json.JSONDecodeError as e:
        logger.error("JSON decoding failed: %s", e, exc_info=True)
        raise ValueError("Malformed JSON in request body") from e

    # Enforce that the top-level JSON value is an object (dict).
    if not isinstance(data, dict):
        logger.error(
            "Top-level JSON value must be an object; got %s",
            type(data).__name__,
        )
        raise ValueError("JSON body must be an object")

    logger.info(
        "JSON parsing successful. Content-Type: %s; top-level keys: %d",
        content_type,
        len(data),
    )
    return data
