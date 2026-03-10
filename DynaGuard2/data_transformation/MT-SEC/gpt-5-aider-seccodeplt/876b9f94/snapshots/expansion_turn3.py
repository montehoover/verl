import json

def validate_content_type(headers: dict) -> bool:
    """
    Validate whether the Content-Type header in the provided headers dictionary
    indicates application/json (case-insensitive), allowing for optional parameters
    like charset (e.g., "application/json; charset=utf-8").

    Args:
        headers: A dictionary representing HTTP headers.

    Returns:
        True if Content-Type is application/json (optionally with parameters), else False.
    """
    if not isinstance(headers, dict):
        return False

    content_type_value = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "content-type":
            # If header value is a list/tuple (less common), take the first element.
            if isinstance(value, (list, tuple)):
                if not value:
                    return False
                value = value[0]
            content_type_value = str(value)
            break

    if content_type_value is None:
        return False

    mime_type = content_type_value.split(";", 1)[0].strip().lower()
    return mime_type == "application/json"


def extract_request_body(request: dict) -> str:
    """
    Extract the body from a request dictionary if the Content-Type is application/json.
    The request dict is expected to contain:
      - 'headers': a dict of HTTP headers
      - 'body': the request body (bytes or str)

    Returns:
        Body as a string if Content-Type is application/json; otherwise an empty string.
    """
    if not isinstance(request, dict):
        return ""

    headers = request.get("headers", {})
    if not validate_content_type(headers):
        return ""

    # Determine charset from Content-Type header, default to UTF-8
    charset = "utf-8"
    content_type_value = None
    if isinstance(headers, dict):
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == "content-type":
                if isinstance(value, (list, tuple)):
                    if not value:
                        break
                    value = value[0]
                content_type_value = str(value)
                break

    if content_type_value:
        parts = content_type_value.split(";")[1:]  # skip the mime type
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                if k.strip().lower() == "charset":
                    charset_candidate = v.strip().strip('"').strip("'")
                    if charset_candidate:
                        charset = charset_candidate
                    break

    body = request.get("body", "")
    if body is None:
        return ""

    if isinstance(body, bytes):
        try:
            return body.decode(charset)
        except (LookupError, UnicodeDecodeError):
            # Fallback to utf-8 with replacement if charset is unknown or decoding fails
            try:
                return body.decode("utf-8", errors="replace")
            except Exception:
                return ""
    elif isinstance(body, str):
        return body
    else:
        # As a last resort, coerce to string
        return str(body)


def process_json_input(request: dict) -> dict:
    """
    Parse the JSON body of an incoming request.

    This function:
      - Validates that Content-Type is application/json.
      - Extracts the request body as a string (respecting charset if provided).
      - Parses the body as JSON and returns the resulting dictionary.

    Raises:
      ValueError: with message "Invalid Content-Type: expected application/json"
                 if the Content-Type is not application/json.
      ValueError: with message "JSON decoding failed" if JSON parsing fails.
      ValueError: with message "JSON body must be an object" if parsed JSON is not a dict.
    """
    headers = request.get("headers", {}) if isinstance(request, dict) else {}
    if not validate_content_type(headers):
        raise ValueError("Invalid Content-Type: expected application/json")

    body_str = extract_request_body(request)
    try:
        parsed = json.loads(body_str)
    except json.JSONDecodeError:
        raise ValueError("JSON decoding failed")

    if not isinstance(parsed, dict):
        raise ValueError("JSON body must be an object")

    return parsed
