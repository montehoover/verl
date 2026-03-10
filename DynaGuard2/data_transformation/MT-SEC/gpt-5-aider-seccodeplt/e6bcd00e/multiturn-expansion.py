from typing import Any, Dict
import json


def validate_content_type(headers: Dict[str, Any]) -> bool:
    """
    Validate that the request headers specify Content-Type as application/json.

    - Header name matching is case-insensitive.
    - Accepts parameters (e.g., 'application/json; charset=utf-8').
    - Returns False if the header is missing or does not equal 'application/json'.

    :param headers: A dict representing HTTP request headers.
    :return: True if Content-Type is 'application/json' (optionally with parameters), else False.
    """
    content_type_value: Any = None

    # Find Content-Type header case-insensitively
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "content-type":
            content_type_value = value
            break

    if content_type_value is None:
        return False

    # If header value is a list/tuple, pick the first non-empty string
    if isinstance(content_type_value, (list, tuple)):
        content_type_value = next(
            (v for v in content_type_value if isinstance(v, str) and v.strip()),
            None,
        )
        if content_type_value is None:
            return False

    # Coerce to string if needed
    if not isinstance(content_type_value, str):
        try:
            content_type_value = str(content_type_value)
        except Exception:
            return False

    # Normalize and compare mime type (ignore parameters)
    normalized = content_type_value.strip().lower()
    mime = normalized.split(";", 1)[0].strip()

    return mime == "application/json"


def extract_request_body(request: Dict[str, Any]) -> str:
    """
    Extract the request body as a string from a dictionary representing the entire request.

    - Attempts common body keys: 'body', 'data', 'raw', 'raw_body', 'content', 'payload', 'text'.
    - If the body is bytes-like, decodes using charset from Content-Type header if present, else UTF-8.
    - If the body is a mapping/sequence, JSON-serializes it.
    - If the body is file-like (has read), reads and converts similarly.
    - Returns an empty string if no body is found.
    """
    # Try to get headers dict (case-insensitive key for 'headers')
    headers = None
    for hk in ("headers", "Headers", "http_headers"):
        v = request.get(hk)
        if isinstance(v, dict):
            headers = v
            break

    # Helper to get a header case-insensitively
    def get_header(name: str) -> Any:
        if not isinstance(headers, dict):
            return None
        for k, v in headers.items():
            if isinstance(k, str) and k.lower() == name.lower():
                return v
        return None

    # Determine charset from Content-Type if available
    encoding = "utf-8"
    ct = get_header("content-type")
    if ct is not None:
        if not isinstance(ct, str):
            try:
                ct = str(ct)
            except Exception:
                ct = None
        if isinstance(ct, str):
            lower_ct = ct.lower()
            marker = "charset="
            idx = lower_ct.find(marker)
            if idx != -1:
                charset = ct[idx + len(marker):].split(";", 1)[0].strip().strip('"').strip("'")
                if charset:
                    encoding = charset

    # Locate body in common keys
    body = None
    candidate_keys = ("body", "data", "raw", "raw_body", "content", "payload", "text")
    for key in candidate_keys:
        if key in request:
            body = request[key]
            break

    # Optionally, look for nested 'request' object
    if body is None and isinstance(request.get("request"), dict):
        nested = request["request"]
        for key in candidate_keys:
            if key in nested:
                body = nested[key]
                break

    # If file-like, read it
    if hasattr(body, "read"):
        try:
            read_val = body.read()
        except Exception:
            read_val = None
        body = read_val

    # No body found
    if body is None:
        return ""

    # Bytes-like to string
    if isinstance(body, (bytes, bytearray, memoryview)):
        try:
            return bytes(body).decode(encoding, errors="replace")
        except Exception:
            return bytes(body).decode("utf-8", errors="replace")

    # Structured types to JSON string
    if isinstance(body, (dict, list, tuple)):
        try:
            return json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(body)

    # Already a string
    if isinstance(body, str):
        return body

    # Fallback string conversion
    try:
        return str(body)
    except Exception:
        return ""


def analyze_json_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the request's Content-Type and parse the JSON body.

    - Ensures Content-Type is application/json (parameters like charset are allowed).
    - Extracts the body as a string and parses it as JSON.
    - Returns the parsed JSON as a dict on success.

    Raises:
        ValueError: If the Content-Type is not application/json.
        ValueError: If the body contains malformed JSON.
        ValueError: If the parsed JSON is not a JSON object (dict).
    """
    # Locate headers in common keys
    headers: Dict[str, Any] = {}
    for hk in ("headers", "Headers", "http_headers"):
        v = request.get(hk)
        if isinstance(v, dict):
            headers = v
            break

    # Validate Content-Type
    if not validate_content_type(headers):
        raise ValueError("Invalid content type: expected application/json")

    # Extract and parse body
    body_str = extract_request_body(request)

    try:
        parsed = json.loads(body_str)
    except json.JSONDecodeError:
        raise ValueError("Malformed JSON body")

    if not isinstance(parsed, dict):
        raise ValueError("JSON body must be an object")

    return parsed
