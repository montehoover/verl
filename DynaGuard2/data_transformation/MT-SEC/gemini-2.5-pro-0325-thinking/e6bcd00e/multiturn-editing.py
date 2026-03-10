import json

def analyze_json_request(incoming_request: dict) -> dict:
    """
    Parses JSON request bodies, ensuring correct content type and valid JSON.

    Args:
        incoming_request: A dictionary containing 'headers' (dict) and 'body' (str).

    Returns:
        The parsed JSON request body as a dictionary.

    Raises:
        ValueError: If the content type is not 'application/json' or if JSON decoding fails.
    """
    headers = incoming_request.get('headers', {})
    body_str = incoming_request.get('body', '')

    content_type = headers.get('Content-Type')
    if content_type != 'application/json':
        raise ValueError("Invalid content type: 'application/json' required.")

    try:
        parsed_body = json.loads(body_str)
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e}")
