import json

def analyze_json_request(incoming_request: dict) -> dict:
    """
    Parse and validate a JSON request body from an incoming request.

    Args:
        incoming_request (dict): Dictionary containing "headers" and "body".

    Returns:
        dict: Parsed JSON object.

    Raises:
        ValueError: If Content-Type is not application/json or if JSON is malformed.
    """
    if not isinstance(incoming_request, dict):
        raise ValueError("Malformed JSON in request body.")

    headers = incoming_request.get("headers") or {}
    normalized_headers = {str(k).lower(): v for k, v in headers.items()} if isinstance(headers, dict) else {}
    content_type = normalized_headers.get("content-type")
    if not content_type:
        raise ValueError("Unsupported Content-Type; expected application/json.")

    mime_type = str(content_type).split(";")[0].strip().lower()
    if mime_type != "application/json":
        raise ValueError("Unsupported Content-Type; expected application/json.")

    body = incoming_request.get("body")

    if isinstance(body, dict):
        return body

    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode("utf-8")
        except Exception:
            raise ValueError("Malformed JSON in request body.")

    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except Exception:
            raise ValueError("Malformed JSON in request body.")
        if not isinstance(parsed, dict):
            raise ValueError("Malformed JSON in request body.")
        return parsed

    raise ValueError("Malformed JSON in request body.")
