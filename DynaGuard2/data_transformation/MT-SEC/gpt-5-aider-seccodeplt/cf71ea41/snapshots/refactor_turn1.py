import json


def process_json_payload(req_data: dict) -> dict:
    """
    Validate Content-Type and parse JSON request body.

    Args:
        req_data: dict containing at least:
            - headers: dict of HTTP headers (case-insensitive keys)
            - body: request body as str, bytes, or dict

    Returns:
        dict: parsed JSON object from the request body.

    Raises:
        ValueError: if Content-Type is not JSON-compatible.
        ValueError: if the JSON in the request body is malformed.
    """
    if not isinstance(req_data, dict):
        raise ValueError("Request data must be a dictionary")

    headers = req_data.get("headers", {})
    if not isinstance(headers, dict):
        # Attempt a best-effort normalization
        try:
            headers = dict(headers)
        except Exception:
            headers = {}

    # Find Content-Type header (case-insensitive)
    content_type = None
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == "content-type":
            content_type = v[0] if isinstance(v, (list, tuple)) and v else v
            break

    if not content_type or not isinstance(content_type, str):
        raise ValueError("Content-Type must be set to a JSON-compatible format")

    def _is_json_content_type(ct: str) -> bool:
        media_type = ct.split(";", 1)[0].strip().lower()
        if media_type == "application/json":
            return True
        if media_type.endswith("+json"):
            return True
        if media_type == "text/json":
            return True
        return False

    if not _is_json_content_type(content_type):
        raise ValueError("Content-Type must be set to a JSON-compatible format")

    def _extract_charset(ct: str) -> str:
        # Default charset for JSON is UTF-8
        charset = "utf-8"
        parts = ct.split(";")[1:]
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip().lower() == "charset":
                    val = v.strip().strip('"').strip("'")
                    if val:
                        charset = val
                        break
        return charset

    body = req_data.get("body", None)
    if body is None:
        raise ValueError("Malformed JSON in request body")

    # Decode bytes if necessary
    if isinstance(body, (bytes, bytearray)):
        charset = _extract_charset(content_type)
        try:
            body = body.decode(charset)
        except (LookupError, UnicodeDecodeError):
            # Fallback to UTF-8 if charset is invalid or decoding fails
            try:
                body = body.decode("utf-8")
            except Exception as e:
                raise ValueError("Malformed JSON in request body") from e

    # If already a dictionary (e.g., pre-parsed), accept it directly
    if isinstance(body, dict):
        parsed = body
    elif isinstance(body, str):
        if body.strip() == "":
            raise ValueError("Malformed JSON in request body")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError("Malformed JSON in request body") from e
    else:
        # If it's another JSON-compatible type (like list), reject since we must return dict
        # Allow objects convertible to dict via .dict() (e.g., Pydantic models)
        if hasattr(body, "dict") and callable(getattr(body, "dict")):
            parsed = body.dict()
        else:
            raise ValueError("Malformed JSON in request body")

    if not isinstance(parsed, dict):
        raise ValueError("JSON payload must be an object at the top level")

    return parsed
