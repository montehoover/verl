import json


def analyze_json_request(incoming_request: dict) -> dict:
    """
    Analyze and parse a JSON request body from an incoming request mapping.

    This function validates that the request has the appropriate Content-Type
    header and that the body contains valid JSON. It accepts bodies provided
    as a str, bytes/bytearray (UTF-8), or already-parsed dict.

    Args:
        incoming_request (dict): A mapping that should include:
            - "headers" (dict): HTTP headers for the request.
            - "body" (str | bytes | bytearray | dict): The request body.

    Returns:
        dict: The parsed JSON object.

    Raises:
        ValueError: If the Content-Type is missing or not application/json.
        ValueError: If the body cannot be decoded or parsed as valid JSON.
        ValueError: If the parsed JSON is not a JSON object (i.e., not a dict).
    """
    # Safely obtain headers; default to an empty dict if missing/invalid.
    headers = incoming_request.get("headers", {})
    if not isinstance(headers, dict):
        headers = {}

    # Locate the Content-Type header in a case-insensitive manner.
    content_type = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "content-type":
            content_type = value
            break

    # Validate that Content-Type is present and of the correct type.
    if not isinstance(content_type, str):
        raise ValueError("Invalid content type. Expected application/json")

    # Normalize the Content-Type header value and allow optional parameters
    # like "application/json; charset=utf-8".
    ct_value = content_type.strip().lower()
    if not (ct_value == "application/json" or
            ct_value.startswith("application/json;")):
        raise ValueError("Invalid content type. Expected application/json")

    # Extract the body for parsing.
    body = incoming_request.get("body", None)

    # If the body is bytes-like, attempt to decode as UTF-8.
    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode("utf-8")
        except Exception:
            # Decoding failed; treat as malformed JSON.
            raise ValueError("Malformed JSON in request body")

    # If the body is already a dict, assume it's valid JSON and return it.
    if isinstance(body, dict):
        return body

    # At this point, we expect a string to parse as JSON.
    if not isinstance(body, str):
        raise ValueError("Malformed JSON in request body")

    # Attempt to parse the JSON string.
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        raise ValueError("Malformed JSON in request body")

    # Ensure the parsed JSON is an object (dict), not an array or primitive.
    if not isinstance(parsed, dict):
        raise ValueError("JSON body must be an object")

    return parsed
