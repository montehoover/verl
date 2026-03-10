import json

def process_json_input(req: dict) -> dict:
    """
    Parses JSON request bodies, ensuring correct content type and valid JSON.

    Args:
        req: A dictionary containing 'headers' and 'body' of the incoming request.
             req['headers'] is a dictionary of HTTP headers.
             req['body'] is a string containing the request body.

    Returns:
        The parsed request body as a dictionary.

    Raises:
        ValueError: If the content type is not 'application/json' or if JSON decoding fails.
    """
    headers = req.get('headers', {})
    body = req.get('body', '')

    content_type = headers.get('Content-Type')
    if content_type != 'application/json':
        raise ValueError("Invalid content type: 'application/json' expected.")

    try:
        parsed_body = json.loads(body)
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e}")
