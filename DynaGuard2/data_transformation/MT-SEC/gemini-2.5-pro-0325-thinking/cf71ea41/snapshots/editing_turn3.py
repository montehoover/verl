import json

def process_json_payload(req_data):
    """
    Parses and validates incoming JSON requests.

    Args:
        req_data (dict): A dictionary containing 'headers' and 'body' of the HTTP request.
                         req_data['headers'] is a dict of HTTP headers.
                         req_data['body'] is a string representing the request body.

    Returns:
        dict: The validated and parsed request body as a dictionary.

    Raises:
        ValueError: If the Content-Type is not 'application/json' or if the JSON body is malformed.
    """
    headers = req_data.get('headers', {})
    body = req_data.get('body', '')

    content_type = headers.get('Content-Type')
    if content_type != 'application/json':
        raise ValueError("Content-Type must be 'application/json'")

    try:
        parsed_body = json.loads(body)
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON payload: {e}")
