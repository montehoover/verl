import json

def process_json_request(req_data: dict) -> dict:
    """
    Parses and validates the body of incoming JSON requests.

    Args:
        req_data: A dictionary containing request headers and body.
                  Expected format: {'headers': {'Content-Type': '...'}, 'body': '...'}

    Returns:
        The parsed and validated request body as a dictionary.

    Raises:
        ValueError: If the content type is not 'application/json' or
                    if there are issues decoding the JSON body.
    """
    headers = req_data.get('headers', {})
    content_type = headers.get('Content-Type', '')

    if content_type.lower() != 'application/json':
        raise ValueError("Invalid content type. Expected 'application/json'.")

    body_str = req_data.get('body', '')
    if not body_str:
        # Handle empty body if it should be considered an error or return empty dict
        # For now, assuming an empty body might be valid for some JSON structures (e.g., "{}")
        # or could be an error if a non-empty JSON is always expected.
        # Let's proceed and let json.loads handle it. If empty string is invalid JSON, it will fail.
        pass

    try:
        parsed_body = json.loads(body_str)
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in request body: {e}")
    except TypeError: # Handles cases where body_str is not a string or bytes-like object
        raise ValueError("Request body must be a string for JSON decoding.")
