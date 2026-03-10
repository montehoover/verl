import json
from typing import Any, Mapping


def check_header_for_json(headers: Mapping[str, Any]) -> bool:
    """
    Return True if the Content-Type header is set to 'application/json', else False.

    - Header name matching is case-insensitive.
    - Allows parameters (e.g., 'application/json; charset=utf-8').
    - If multiple values are provided as a list/tuple, returns True if any match.
    """
    content_type_value: Any | None = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "content-type":
            content_type_value = value
            break

    if content_type_value is None:
        return False

    # If the value can be a list/tuple of header values, check any of them.
    if isinstance(content_type_value, (list, tuple)):
        for v in content_type_value:
            if isinstance(v, str):
                media_type = v.strip().lower().split(";", 1)[0]
                if media_type == "application/json":
                    return True
        return False

    # Coerce non-strings to string (defensive).
    if not isinstance(content_type_value, str):
        content_type_value = str(content_type_value)

    media_type = content_type_value.strip().lower().split(";", 1)[0]
    return media_type == "application/json"


def decode_json_body(body: str) -> dict[str, Any]:
    """
    Decode a JSON-encoded string into a Python dict.

    Raises:
        ValueError: If the JSON is invalid or if the top-level value is not an object.
    """
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg}") from e

    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")

    return data


def validate_json_request(input_data: Mapping[str, Any]) -> dict[str, Any]:
    """
    Validate an incoming request-like mapping that contains 'headers' and 'body'.

    - Ensures Content-Type is 'application/json' (case-insensitive, parameters allowed).
    - Parses the body as JSON and ensures the top-level value is an object.
    - Returns the parsed JSON object (dict) on success.

    Raises:
        ValueError: If Content-Type is not application/json, the body is missing or not a string/bytes,
                    or if the JSON is invalid or not an object.
    """
    if not isinstance(input_data, Mapping):
        raise ValueError("input_data must be a mapping with 'headers' and 'body'")

    headers = input_data.get("headers")
    if not isinstance(headers, Mapping):
        raise ValueError("headers must be provided as a mapping")

    if not check_header_for_json(headers):
        raise ValueError("Content-Type must be 'application/json'")

    body = input_data.get("body")

    if isinstance(body, bytes):
        try:
            body_str = body.decode("utf-8")
        except Exception:
            raise ValueError("Request body must be a UTF-8 JSON-encoded string")
    elif isinstance(body, str):
        body_str = body
    else:
        raise ValueError("Request body must be a JSON-encoded string")

    return decode_json_body(body_str)
