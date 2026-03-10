import json
from typing import Any, Dict, Iterable, Tuple, Union


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

    # Normalize headers into a case-insensitive dict
    raw_headers = input_data.get("headers", {})
    headers: Dict[str, str] = {}

    if isinstance(raw_headers, dict):
        for k, v in raw_headers.items():
            headers[str(k).lower()] = v
    elif isinstance(raw_headers, Iterable):
        try:
            for k, v in raw_headers:  # type: ignore[misc]
                headers[str(k).lower()] = v
        except Exception:
            headers = {}
    else:
        headers = {}

    # Validate Content-Type
    content_type = headers.get("content-type") or headers.get("content_type")
    if not content_type or not isinstance(content_type, str):
        raise ValueError("Unsupported Content-Type.")

    media_type = content_type.split(";", 1)[0].strip().lower()
    is_json_ct = (
        media_type == "application/json"
        or (media_type.startswith("application/") and media_type.split("/", 1)[1].endswith("+json"))
    )
    if not is_json_ct:
        raise ValueError("Unsupported Content-Type.")

    # Extract and normalize body to text
    body: Union[str, bytes, None] = input_data.get("body")
    if body is None:
        raise ValueError("Invalid JSON: request body is empty.")

    if isinstance(body, bytes):
        try:
            body_text = body.decode("utf-8")
        except Exception:
            # Fallback to strict error message for non-decodable payloads
            raise ValueError("Invalid JSON: could not decode request body as UTF-8.")
    elif isinstance(body, str):
        body_text = body
    else:
        # Unsupported body type
        raise ValueError("Invalid JSON: unsupported body type.")

    if body_text.strip() == "":
        raise ValueError("Invalid JSON: request body is empty.")

    # Parse JSON
    try:
        data: Any = json.loads(body_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg}") from None

    if not isinstance(data, dict):
        raise ValueError("Invalid JSON: top-level value must be an object.")

    return data
