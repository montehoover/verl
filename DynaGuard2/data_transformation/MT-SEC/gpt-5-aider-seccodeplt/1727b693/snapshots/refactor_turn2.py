import json
from typing import Any, Dict, Iterable, Union


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
        raise ValueError("Invalid request data.")

    # Normalize headers to a case-insensitive dict for easy lookups.
    raw_headers = input_data.get("headers", {})
    headers: Dict[str, str] = {}

    if isinstance(raw_headers, dict):
        for key, value in raw_headers.items():
            headers[str(key).lower()] = value
    elif isinstance(raw_headers, Iterable):
        # Allow an iterable of (key, value) header pairs.
        try:
            for key, value in raw_headers:  # type: ignore[misc]
                headers[str(key).lower()] = value
        except Exception:
            # If headers cannot be iterated as pairs, fall back to empty.
            headers = {}
    else:
        headers = {}

    # Validate Content-Type header:
    # - Must be present
    # - Must be "application/json" or "application/*+json"
    content_type = headers.get("content-type") or headers.get("content_type")
    if not content_type or not isinstance(content_type, str):
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
        raise ValueError("Unsupported Content-Type.")

    # Extract the body and normalize it to a text string.
    body: Union[str, bytes, None] = input_data.get("body")
    if body is None:
        raise ValueError("Invalid JSON: request body is empty.")

    if isinstance(body, bytes):
        # Attempt UTF-8 decoding, which is the default for JSON.
        try:
            body_text = body.decode("utf-8")
        except Exception:
            raise ValueError(
                "Invalid JSON: could not decode request body as UTF-8."
            )
    elif isinstance(body, str):
        body_text = body
    else:
        # Body must be a str or bytes for JSON decoding.
        raise ValueError("Invalid JSON: unsupported body type.")

    if body_text.strip() == "":
        raise ValueError("Invalid JSON: request body is empty.")

    # Decode JSON. Any parse error should surface as a clear ValueError.
    try:
        data: Any = json.loads(body_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from None

    # The top-level JSON value must be an object for request bodies.
    if not isinstance(data, dict):
        raise ValueError("Invalid JSON: top-level value must be an object.")

    return data
