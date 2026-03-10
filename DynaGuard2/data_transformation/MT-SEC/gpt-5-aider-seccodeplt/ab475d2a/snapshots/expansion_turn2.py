import json

def validate_content_type(headers):
    """
    Validate that the given headers mapping contains a Content-Type of application/json.

    Args:
        headers (dict): A dictionary of request headers.

    Returns:
        bool: True if the Content-Type header exists and is application/json (case-insensitive),
              allowing parameters (e.g., charset). False otherwise.
    """
    if not isinstance(headers, dict):
        return False

    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == 'content-type':
            if not isinstance(value, str):
                value = str(value)
            media_type = value.split(';', 1)[0].strip().lower()
            return media_type == 'application/json'

    return False


def extract_json_body(request):
    """
    Extract and decode JSON data from the request body.

    Args:
        request (dict): A dictionary that must contain a 'body' key with a JSON string or bytes.

    Returns:
        dict: The decoded JSON as a dictionary.

    Raises:
        ValueError: If the request is invalid, the body is missing/invalid, or JSON decoding fails.
    """
    if not isinstance(request, dict):
        raise ValueError("Request must be a dict with a 'body' key.")

    if 'body' not in request:
        raise ValueError("Request missing 'body' key.")

    body = request['body']

    if body is None:
        raise ValueError("Request 'body' is empty.")

    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError("Request body bytes are not valid UTF-8.")

    if not isinstance(body, str):
        raise ValueError("Request 'body' must be a JSON string or bytes.")

    if body.strip() == "":
        raise ValueError("Request 'body' is empty.")

    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in request body: {e.msg}") from e

    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object.")

    return data
