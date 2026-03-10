import json

def decode_json_request(req: dict) -> dict:
    """
    Validate that the request has Content-Type: application/json and decode the JSON body.
    Raises:
        ValueError: with a broad message if Content-Type is invalid.
        ValueError: with a specific message if JSON format is invalid.
    """
    # Validate request structure
    if not isinstance(req, dict):
        raise ValueError("Invalid content type.")

    headers = req.get('headers')
    if not isinstance(headers, dict):
        raise ValueError("Invalid content type.")

    # Find Content-Type header (case-insensitive)
    content_type_value = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == 'content-type':
            content_type_value = value
            break

    if content_type_value is None:
        raise ValueError("Invalid content type.")

    # Normalize header value to string
    if isinstance(content_type_value, (list, tuple)):
        if not content_type_value:
            raise ValueError("Invalid content type.")
        content_type_value = content_type_value[0]

    if isinstance(content_type_value, (bytes, bytearray)):
        try:
            content_type_value = content_type_value.decode('ascii', 'strict')
        except Exception:
            raise ValueError("Invalid content type.")

    if not isinstance(content_type_value, str):
        raise ValueError("Invalid content type.")

    media_type = content_type_value.split(';', 1)[0].strip().lower()
    if media_type != 'application/json':
        raise ValueError("Invalid content type.")

    # Decode body
    body = req.get('body')

    # Accept already-parsed dict
    if isinstance(body, dict):
        return body

    # Convert bytes to string
    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode('utf-8', 'strict')
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}") from None

    if not isinstance(body, str):
        raise ValueError("Invalid JSON format: expected a JSON string or object.")

    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e.msg}") from None

    if not isinstance(data, dict):
        raise ValueError("Invalid JSON format: expected a JSON object.")

    return data
