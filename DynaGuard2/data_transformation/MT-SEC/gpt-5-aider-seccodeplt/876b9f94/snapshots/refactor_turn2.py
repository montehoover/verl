"""
Utilities for parsing and validating JSON request bodies.

This module provides a helper function to parse JSON payloads from a
FastAPI-style request dictionary. It enforces that the Content-Type header
indicates a JSON payload and that the body is valid JSON representing an
object (i.e., a top-level dictionary).
"""

import json


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
        raise ValueError(
            "Request must be a dictionary containing 'headers' and 'body'"
        )

    # Extract and validate headers.
    headers = req.get("headers", {})
    if not isinstance(headers, dict):
        raise ValueError("Request headers must be a dictionary")

    # Locate Content-Type header (case-insensitive).
    content_type = None
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == "content-type":
            content_type = v
            break

    # Ensure Content-Type is present and valid.
    if not content_type:
        raise ValueError("Missing Content-Type header; expected 'application/json'")
    if not isinstance(content_type, str):
        raise ValueError("Invalid Content-Type header value; expected a string")

    # Parse media type and optional parameters (e.g., charset).
    parts = [p.strip() for p in content_type.split(";") if p.strip()]
    media_type = parts[0].lower() if parts else ""
    if media_type != "application/json":
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

    # Retrieve the body.
    body = req.get("body", None)

    # If body is already a dict, accept it as parsed JSON.
    if isinstance(body, dict):
        return body

    # Convert body to a string if needed.
    if isinstance(body, (bytes, bytearray, memoryview)):
        try:
            body_str = bytes(body).decode(charset)
        except (LookupError, UnicodeDecodeError) as e:
            # LookupError -> unknown charset; UnicodeDecodeError -> decoding failure.
            raise ValueError(f"Invalid character encoding for request body: {e}") from e
    elif isinstance(body, str):
        body_str = body
    elif body is None:
        # Treat missing body as malformed JSON as per contract.
        raise ValueError("Malformed JSON in request body")
    else:
        # Unsupported body type.
        raise ValueError("Request body must be a JSON string or bytes")

    # Parse JSON into a Python object.
    try:
        data = json.loads(body_str)
    except json.JSONDecodeError as e:
        raise ValueError("Malformed JSON in request body") from e

    # Enforce that the top-level JSON value is an object (dict).
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")

    return data
