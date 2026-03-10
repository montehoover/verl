import json

def validate_json_request(input_data: dict) -> dict:
    """
    Parses and validates the body of incoming JSON requests.

    Args:
        input_data: A dictionary containing 'headers' and 'body' of the request.
                    'headers' is a dict, and 'body' is expected to be a JSON string.

    Returns:
        A dictionary representing the parsed and validated request body.

    Raises:
        ValueError: If the content type is not 'application/json' or
                    if there are issues with JSON decoding.
    """
    headers = input_data.get('headers', {})
    content_type = headers.get('Content-Type', '')

    if 'application/json' not in content_type.lower():
        raise ValueError("Invalid content type. Expected 'application/json'.")

    request_body_str = input_data.get('body')
    if request_body_str is None:
        raise ValueError("Request body is missing.")

    try:
        # Assuming body is a string that needs to be decoded
        if isinstance(request_body_str, bytes):
            request_body_str = request_body_str.decode('utf-8')
        
        parsed_body = json.loads(request_body_str)
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        # Catch any other unexpected errors during body processing
        raise ValueError(f"Error processing request body: {e}")
