import json


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
        headers = {}

    headers_lc = {str(k).lower(): v for k, v in headers.items()}
    content_type_raw = str(headers_lc.get("content-type", "")).strip()

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

    acceptable = mime_type == "application/json" or (
        mime_type.startswith("application/") and mime_type.endswith("+json")
    )
    if not acceptable:
        raise ValueError("Request content type is not acceptable.")

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
    if isinstance(body, dict):
        return body

    if body is None:
        raise ValueError("Invalid JSON format: request body is empty.")

    if isinstance(body, bytes):
        try:
            body_text = body.decode(charset, errors="strict")
        except (LookupError, UnicodeDecodeError):
            raise ValueError("Invalid JSON format: unable to decode request body.")
    elif isinstance(body, str):
        body_text = body
    else:
        raise ValueError("Invalid JSON format: unsupported body type.")

    body_text = body_text.strip()
    if not body_text:
        raise ValueError("Invalid JSON format: request body is empty.")

    try:
        parsed = json.loads(body_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e.msg}") from None

    if not isinstance(parsed, dict):
        raise ValueError("Invalid JSON format: top-level JSON value must be an object.")

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
    if not isinstance(req_data, dict):
        raise ValueError("Invalid request data.")

    headers = req_data.get("headers") or {}
    charset = validate_json_content_type(headers)

    body = req_data.get("body", None)
    return decode_json_body(body, charset)
