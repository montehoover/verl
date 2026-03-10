import json

def validate_json_request(input_data: dict) -> dict:
    """
    Validates a request for JSON content type and parses the JSON body.

    Checks if the 'Content-Type' header is 'application/json' and
    parses the 'body' of the input_data as a JSON string.

    Args:
        input_data: A dictionary expected to contain 'headers' (a dict)
                    and 'body' (a JSON string).

    Returns:
        A Python dictionary parsed from the JSON string in the body.

    Raises:
        ValueError: If 'headers' or 'body' are missing,
                    if 'Content-Type' header is not 'application/json',
                    or if JSON parsing fails.
    """
    headers = input_data.get('headers')
    if not headers or not isinstance(headers, dict):
        raise ValueError("Missing or invalid 'headers' in input_data.")

    content_type = headers.get('Content-Type')
    if content_type != 'application/json':
        raise ValueError(f"Invalid Content-Type: expected 'application/json', got '{content_type}'.")

    body = input_data.get('body')
    if body is None: # body could be an empty string, which is valid JSON for ""
        raise ValueError("Missing 'body' in input_data.")
    
    if not isinstance(body, str):
        raise ValueError("Request 'body' must be a string.")

    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in request body: {e}")
