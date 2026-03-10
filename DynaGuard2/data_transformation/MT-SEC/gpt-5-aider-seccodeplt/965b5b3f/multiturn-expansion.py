import json

def validate_content_type(headers: dict) -> bool:
    """
    Validate that the incoming request headers specify Content-Type: application/json.

    - Header name matching is case-insensitive.
    - Parameters like `; charset=utf-8` are allowed.
    """
    if not isinstance(headers, dict):
        return False

    content_type_value = None
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == "content-type":
            content_type_value = v
            break

    if content_type_value is None:
        return False

    # If value is list-like, use the first element.
    if isinstance(content_type_value, (list, tuple)):
        content_type_value = content_type_value[0] if content_type_value else ""

    if not isinstance(content_type_value, str):
        # Try to coerce to string
        try:
            content_type_value = str(content_type_value)
        except Exception:
            return False

    media_type = content_type_value.split(";")[0].strip().lower()

    return media_type == "application/json"


def decode_json_body(body: str) -> dict:
    """
    Decode a JSON string into a Python dictionary.

    Raises:
        ValueError: If the JSON is malformed or does not decode to an object/dictionary.
    """
    if not isinstance(body, str):
        raise ValueError("JSON body must be a string.")

    # Allow and strip optional UTF-8 BOM
    body = body.lstrip("\ufeff")

    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e.msg} (line {e.lineno} column {e.colno} char {e.pos})") from e

    if not isinstance(data, dict):
        raise ValueError("JSON body must decode to an object/dictionary.")

    return data


def process_json_request(req_data: dict) -> dict:
    """
    Process an incoming JSON request represented by a dictionary containing
    'headers' and 'body'. Validates the Content-Type and decodes the JSON body.

    Args:
        req_data: A dictionary with keys:
            - 'headers': mapping of request headers
            - 'body': string containing the JSON payload

    Returns:
        dict: The parsed JSON object.

    Raises:
        ValueError: If Content-Type is not application/json, the request data is
                    not a dictionary, or the JSON is malformed.
    """
    if not isinstance(req_data, dict):
        raise ValueError("Request data must be a dictionary.")

    headers = req_data.get("headers", {})
    body = req_data.get("body", "")

    if not validate_content_type(headers):
        raise ValueError("Invalid Content-Type: application/json required.")

    return decode_json_body(body)
