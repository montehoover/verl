import json


def process_json_request(req_data: dict) -> dict:
    """
    Parse and validate a JSON request.

    Args:
        req_data: dict with keys:
            - headers: dict of request headers
            - body: the request body (str, bytes, or dict)

    Returns:
        dict: Parsed JSON object.

    Raises:
        ValueError: If content type is not acceptable or JSON is invalid.
    """
    if not isinstance(req_data, dict):
        raise ValueError("Invalid JSON format: request data must be a dictionary")

    headers = req_data.get("headers") or {}
    if not isinstance(headers, dict):
        headers = {}

    # Normalize header keys to lower-case for case-insensitive lookup
    headers_lower = {str(k).lower(): v for k, v in headers.items()}
    content_type = headers_lower.get("content-type")
    if not isinstance(content_type, str) or not content_type.strip():
        raise ValueError("Unsupported content type")

    media_type = content_type.split(";")[0].strip().lower()
    if media_type != "application/json":
        raise ValueError("Unsupported content type")

    body = req_data.get("body")

    # Handle body types
    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode("utf-8")
        except Exception:
            raise ValueError("Invalid JSON format: cannot decode request body as UTF-8")

    if isinstance(body, dict):
        return body

    if body is None or (isinstance(body, str) and body.strip() == ""):
        raise ValueError("Invalid JSON format: request body is empty")

    if not isinstance(body, str):
        # Fallback: attempt to serialize to string for JSON parsing
        body = str(body)

    try:
        data = json.loads(body)
    except Exception as e:
        raise ValueError(f"Invalid JSON format: {e}") from None

    if not isinstance(data, dict):
        raise ValueError("Invalid JSON format: JSON body must be an object")

    return data
