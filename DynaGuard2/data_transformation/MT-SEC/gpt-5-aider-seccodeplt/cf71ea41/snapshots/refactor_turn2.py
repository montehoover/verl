"""
Utilities for parsing and validating JSON HTTP request payloads.

This module provides a helper function, `process_json_payload`, which validates
the Content-Type header and parses a JSON request body into a dictionary.
"""

import json
from typing import Any, Dict, Mapping, Optional


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


def process_json_payload(req_data: dict) -> dict:
    """
    Validate Content-Type and parse a JSON request body.

    This function validates that the request's Content-Type indicates a JSON
    payload and attempts to parse the body into a Python dictionary. It accepts
    a body in str, bytes/bytearray, or dict form. If the JSON is malformed or
    the Content-Type is not JSON-compatible, a ValueError is raised.

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
    # Validate the overall structure of req_data.
    if not isinstance(req_data, dict):
        raise ValueError("Request data must be a dictionary")

    # Normalize headers to a dictionary-like structure.
    headers: Dict[str, Any] = req_data.get("headers", {})  # type: ignore[assignment]
    if not isinstance(headers, dict):
        try:
            headers = dict(headers)  # type: ignore[arg-type]
        except Exception:
            headers = {}

    # Extract and validate the Content-Type header.
    content_type_raw = _get_content_type_header(headers)
    if not content_type_raw or not isinstance(content_type_raw, str):
        raise ValueError("Content-Type must be set to a JSON-compatible format")

    if not _is_json_content_type(content_type_raw):
        raise ValueError("Content-Type must be set to a JSON-compatible format")

    # Retrieve the request body.
    body = req_data.get("body", None)
    if body is None:
        # No body provided is considered malformed for this context.
        raise ValueError("Malformed JSON in request body")

    # Decode bytes using the charset from Content-Type if necessary.
    if isinstance(body, (bytes, bytearray)):
        charset = _extract_charset(content_type_raw)
        try:
            body = body.decode(charset)
        except (LookupError, UnicodeDecodeError):
            # Fallback to UTF-8 if the provided charset is invalid or decoding fails.
            try:
                body = body.decode("utf-8")
            except Exception as exc:
                raise ValueError("Malformed JSON in request body") from exc

    # Parse or normalize the body to a dictionary.
    if isinstance(body, dict):
        parsed = body
    elif isinstance(body, str):
        if body.strip() == "":
            raise ValueError("Malformed JSON in request body")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed JSON in request body") from exc
    else:
        # Support objects that can produce a dictionary via a .dict() method
        # (e.g., Pydantic models). Otherwise, reject as malformed.
        if hasattr(body, "dict") and callable(getattr(body, "dict")):
            parsed = body.dict()
        else:
            raise ValueError("Malformed JSON in request body")

    # Ensure the top-level JSON value is an object.
    if not isinstance(parsed, dict):
        raise ValueError("JSON payload must be an object at the top level")

    return parsed
