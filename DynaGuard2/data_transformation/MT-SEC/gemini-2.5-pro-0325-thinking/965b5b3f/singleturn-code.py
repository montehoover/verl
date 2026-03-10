import json

def process_json_request(req_data: dict) -> dict:
    """
    Parses and validates the body of incoming JSON requests.

    Args:
        req_data: A dictionary containing the request headers and body.
                  Expected format: {"headers": {"Content-Type": "application/json"}, "body": "json_string"}

    Returns:
        The parsed and validated request body as a dictionary.

    Raises:
        ValueError: If the content type is not 'application/json'.
        ValueError: If there is an issue decoding the JSON body.
    """
    headers = req_data.get("headers", {})
    content_type = headers.get("Content-Type")

    if content_type != "application/json":
        raise ValueError("Invalid content type. Expected 'application/json'.")

    body_str = req_data.get("body")
    if body_str is None:
        raise ValueError("Request body is missing.")

    try:
        parsed_body = json.loads(body_str)
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in request body: {e}")
