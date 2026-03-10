"""
Utilities for parsing and validating JSON HTTP request payloads.

This module provides a helper function, `process_json_payload`, which validates
the Content-Type header and parses a JSON request body into a dictionary.
"""

import json
import logging
from typing import Any, Dict, Mapping, Optional


SENSITIVE_HEADERS = {"authorization", "proxy-authorization", "cookie", "set-cookie"}


def _is_json_content_type(content_type: str) -> bool:
    """
    Determine if a Content-Type header value is JSON-compatible.

    Accepts canonical application/json, structured syntax suffixes (+json),
    and a few common variants such as text/json.

    Args:
        content_type: Raw Content-Type header value.

    Returns:
        True if the media type indicates JSON; otherwise, False.
    """
    media_type = content_type.split(";", 1)[0].strip().lower()

    if media_type == "application/json":
        return True

    if media_type.endswith("+json"):
        return True

    if media_type == "text/json":
        return True

    return False


def _extract_charset(content_type: str) -> str:
    """
    Extract the charset from a Content-Type header value.

    Defaults to UTF-8 if not provided.

    Args:
        content_type: Raw Content-Type header value.

    Returns:
        The charset string (e.g., "utf-8").
    """
    # Default charset for JSON per RFC 8259 is UTF-8.
    charset = "utf-8"

    # Parse parameters after the media type.
    parts = content_type.split(";")[1:]
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            if key.strip().lower() == "charset":
                val = value.strip().strip('"').strip("'")
                if val:
                    charset = val
                    break

    return charset


def _get_content_type_header(headers: Mapping[str, Any]) -> Optional[str]:
    """
    Retrieve the Content-Type header value from a headers mapping.

    Performs a case-insensitive lookup and normalizes list/tuple values by
    selecting the first element.

    Args:
        headers: Mapping of HTTP header names to values.

    Returns:
        The Content-Type header value as a string if present, otherwise None.
    """
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "content-type":
            if isinstance(value, (list, tuple)) and value:
                return value[0]
            return value  # Could be str or other type.
    return None


def _truncate_text(text: str, limit: int = 1000) -> str:
    """
    Truncate a string to a maximum length for logging purposes.

    Args:
        text: The string to truncate.
        limit: Maximum number of characters.

    Returns:
        The possibly truncated string with an indicator if truncated.
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


def _normalize_headers_for_logging(headers: Mapping[str, Any]) -> Dict[str, str]:
    """
    Normalize headers to a string dictionary suitable for logging.

    Redacts sensitive headers and truncates long values.

    Args:
        headers: Mapping of HTTP headers.

    Returns:
        A dictionary with stringified and possibly redacted header values.
    """
    normalized: Dict[str, str] = {}
    for key, value in headers.items():
        key_str = str(key)
        if key_str.lower() in SENSITIVE_HEADERS:
            normalized[key_str] = "<redacted>"
            continue

        try:
            if isinstance(value, (list, tuple)):
                value_str = ", ".join(map(str, value))
            else:
                value_str = str(value)
        except Exception:
            value_str = "<unrepresentable>"

        normalized[key_str] = _truncate_text(value_str, 500)

    return normalized


def process_json_payload(req_data: dict) -> dict:
    """
    Validate Content-Type and parse a JSON request body.

    This function validates that the request's Content-Type indicates a JSON
    payload and attempts to parse the body into a Python dictionary. It accepts
    a body in str, bytes/bytearray, or dict form. If the JSON is malformed or
    the Content-Type is not JSON-compatible, a ValueError is raised.

    Logging:
        Initializes a module-specific logger and emits DEBUG-level messages
        describing each processing step, including normalized headers and a
        truncated preview of the request body.

    Args:
        req_data: Dictionary including both the headers and body of the HTTP
            request. Expected keys:
            - "headers": dict-like collection of HTTP headers (case-insensitive)
            - "body": request payload as str, bytes/bytearray, or dict

    Returns:
        dict: Contains the validated and parsed request body.

    Raises:
        ValueError: If the Content-Type is not set to a JSON-compatible format.
        ValueError: If the JSON in the request body is malformed.
    """
    # Initialize logger locally as requested; avoid altering the root logger.
    logger = logging.getLogger("json_payload_processor")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    logger.debug("Starting process_json_payload")

    # Validate the overall structure of req_data.
    if not isinstance(req_data, dict):
        logger.error(
            "Invalid req_data type: %s. Expected dict.", type(req_data).__name__
        )
        raise ValueError("Request data must be a dictionary")

    # Normalize headers to a dictionary-like structure.
    headers: Dict[str, Any] = req_data.get("headers", {})  # type: ignore[assignment]
    if not isinstance(headers, dict):
        try:
            headers = dict(headers)  # type: ignore[arg-type]
            logger.debug("Headers converted to dict from mapping-like object.")
        except Exception:
            logger.debug(
                "Headers could not be converted to dict; defaulting to empty dict."
            )
            headers = {}

    logger.debug("Received headers (normalized): %s", _normalize_headers_for_logging(headers))

    # Extract and validate the Content-Type header.
    content_type_raw = _get_content_type_header(headers)
    logger.debug("Content-Type header raw value: %r", content_type_raw)

    if not content_type_raw or not isinstance(content_type_raw, str):
        logger.error("Missing or invalid Content-Type header.")
        raise ValueError("Content-Type must be set to a JSON-compatible format")

    if not _is_json_content_type(content_type_raw):
        logger.error("Non-JSON Content-Type detected: %r", content_type_raw)
        raise ValueError("Content-Type must be set to a JSON-compatible format")

    # Retrieve the request body.
    body = req_data.get("body", None)
    if body is None:
        logger.error("Request body is missing (None).")
        raise ValueError("Malformed JSON in request body")

    logger.debug("Initial body type: %s", type(body).__name__)

    # Decode bytes using the charset from Content-Type if necessary.
    if isinstance(body, (bytes, bytearray)):
        charset = _extract_charset(content_type_raw)
        logger.debug("Detected charset for decoding: %s", charset)
        try:
            body = body.decode(charset)
            logger.debug(
                "Body decoded using charset %s. Preview: %s",
                charset,
                _truncate_text(body, 1000),
            )
        except (LookupError, UnicodeDecodeError):
            logger.warning(
                "Decoding with charset %s failed; attempting UTF-8 fallback.",
                charset,
            )
            try:
                body = body.decode("utf-8")
                logger.debug(
                    "Body decoded using UTF-8 fallback. Preview: %s",
                    _truncate_text(body, 1000),
                )
            except Exception as exc:
                logger.exception("Failed to decode request body as text.")
                raise ValueError("Malformed JSON in request body") from exc

    # Parse or normalize the body to a dictionary.
    if isinstance(body, dict):
        parsed = body
        try:
            body_preview = _truncate_text(json.dumps(parsed), 1000)
        except Exception:
            body_preview = "<unserializable dict>"
        logger.debug("Body is already a dict. Preview JSON: %s", body_preview)
    elif isinstance(body, str):
        if body.strip() == "":
            logger.error("Request body is empty or whitespace only.")
            raise ValueError("Malformed JSON in request body")
        try:
            logger.debug("Parsing JSON body. Preview: %s", _truncate_text(body, 1000))
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.exception("JSON decoding failed for request body.")
            raise ValueError("Malformed JSON in request body") from exc
    else:
        # Support objects that can produce a dictionary via a .dict() method
        # (e.g., Pydantic models). Otherwise, reject as malformed.
        if hasattr(body, "dict") and callable(getattr(body, "dict")):
            try:
                parsed = body.dict()
                logger.debug(
                    "Body converted via .dict(). Preview: %s",
                    _truncate_text(json.dumps(parsed), 1000),
                )
            except Exception as exc:
                logger.exception("Failed to convert body via .dict().")
                raise ValueError("Malformed JSON in request body") from exc
        else:
            logger.error(
                "Unsupported body type: %s; cannot parse as JSON.",
                type(body).__name__,
            )
            raise ValueError("Malformed JSON in request body")

    # Ensure the top-level JSON value is an object.
    if not isinstance(parsed, dict):
        logger.error(
            "Top-level JSON value is not an object (type=%s).",
            type(parsed).__name__,
        )
        raise ValueError("JSON payload must be an object at the top level")

    logger.debug("Successfully parsed JSON body with %d keys.", len(parsed))
    return parsed
