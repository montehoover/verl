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
    # Retrieve headers from input_data.
    # Defaults to an empty dictionary if 'headers' key is not present.
    headers = input_data.get('headers', {})
    # Extract the Content-Type header.
    # Defaults to an empty string if 'Content-Type' is not in headers.
    content_type = headers.get('Content-Type', '')

    # Validate the content type.
    # The request must have a 'Content-Type' header indicating 'application/json'.
    # The check is performed in a case-insensitive manner.
    if 'application/json' not in content_type.lower():
        raise ValueError("Invalid content type. Expected 'application/json'.")

    # Retrieve the raw request body.
    request_body_str = input_data.get('body')
    if request_body_str is None:
        # Ensure the request body is not missing.
        raise ValueError("Request body is missing.")

    try:
        # Decode the request body if it's in bytes.
        # HTTP request bodies are often received as byte streams and need decoding
        # (commonly to UTF-8) before JSON parsing.
        if isinstance(request_body_str, bytes):
            request_body_str = request_body_str.decode('utf-8')
        
        # Parse the JSON string.
        # This converts the JSON string into a Python dictionary.
        parsed_body = json.loads(request_body_str)
        return parsed_body
    except json.JSONDecodeError as e:
        # Handle JSON decoding errors.
        # This occurs if the request body is not valid JSON.
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        # Handle other potential errors during body processing.
        # This is a catch-all for unexpected issues.
        raise ValueError(f"Error processing request body: {e}")
