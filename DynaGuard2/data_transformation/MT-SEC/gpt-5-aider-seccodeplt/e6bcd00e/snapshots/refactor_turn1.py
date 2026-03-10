import json


def analyze_json_request(incoming_request: dict) -> dict:
    """
    Parse and validate a JSON request body from an incoming request dictionary.

    The incoming_request should contain:
      - headers (dict): HTTP headers for the request.
      - body (str | bytes | dict): The raw request body.

    Returns:
        dict: The parsed JSON object.

    Raises:
        ValueError: If the Content-Type is not application/json or the JSON is malformed.
    """
    headers = incoming_request.get("headers", {})
    if not isinstance(headers, dict):
        headers = {}

    # Find Content-Type header (case-insensitive)
    content_type = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "content-type":
            content_type = value
            break

    if not isinstance(content_type, str):
        raise ValueError("Invalid content type. Expected application/json")

    ct_value = content_type.strip().lower()
    # Accept parameters like "application/json; charset=utf-8"
    if not (ct_value == "application/json" or ct_value.startswith("application/json;")):
        raise ValueError("Invalid content type. Expected application/json")

    body = incoming_request.get("body", None)

    # If body is bytes, try to decode as UTF-8
    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode("utf-8")
        except Exception:
            raise ValueError("Malformed JSON in request body")

    if isinstance(body, dict):
        return body

    if not isinstance(body, str):
        # Non-string body cannot be parsed as JSON
        raise ValueError("Malformed JSON in request body")

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        raise ValueError("Malformed JSON in request body")

    if not isinstance(parsed, dict):
        raise ValueError("JSON body must be an object")

    return parsed
