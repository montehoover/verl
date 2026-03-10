import json

def _validate_content_type(headers: dict) -> None:
    """
    Validates that the Content-Type header is 'application/json'.

    Args:
        headers: A dictionary of request headers.

    Raises:
        ValueError: If the content type is not 'application/json'.
    """
    content_type = headers.get('Content-Type', '')
    if content_type.lower() != 'application/json':
        raise ValueError("Invalid content type. Expected 'application/json'.")

def _decode_json_body(body_str: str) -> dict:
    """
    Decodes a JSON string into a dictionary.

    Args:
        body_str: The JSON string to decode.

    Returns:
        The decoded dictionary.

    Raises:
        ValueError: If there are issues decoding the JSON body or if the body
                    is not a string.
    """
    # The 'if not body_str: pass' was here.
    # json.loads('') will raise JSONDecodeError, so explicit check isn't strictly needed
    # unless empty string should be treated as, e.g., empty dict.
    # Current behavior: empty string body will raise ValueError via JSONDecodeError.
    try:
        parsed_body = json.loads(body_str)
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in request body: {e}")
    except TypeError: # Handles cases where body_str is not a string or bytes-like object
        raise ValueError("Request body must be a string for JSON decoding.")

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
    _validate_content_type(headers)

    body_str = req_data.get('body', '')
    parsed_body = _decode_json_body(body_str)
    
    return parsed_body
