import json

def validate_json_request(input_data: dict) -> dict:
    """
    Parses and validates the body of incoming JSON requests.

    Args:
        input_data: A dictionary containing 'headers' and 'body' of the request.
                    'headers' is a dict, and 'body' is a JSON string.

    Returns:
        The parsed and validated request body as a dictionary.

    Raises:
        ValueError: If the content type is not 'application/json' or
                    if there are issues with the JSON format.
    """
    headers = input_data.get("headers", {})
    content_type = headers.get("Content-Type")

    if content_type != "application/json":
        raise ValueError("Invalid content type. Expected 'application/json'.")

    body_str = input_data.get("body")
    if body_str is None:
        raise ValueError("Request body is missing.")

    try:
        parsed_body = json.loads(body_str)
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
