import json

def decode_json_request(req):
    """
    Processes incoming JSON requests.

    Validates that the content type is 'application/json' and then decodes
    the JSON body, returning it as a structured dictionary.

    Args:
        req (dict): A dictionary containing 'headers' (dict) and 'body' (str).

    Returns:
        dict: The decoded JSON body.

    Raises:
        ValueError: If the content type is incorrect or if there's an issue
                    with the JSON format.
    """
    headers = req.get('headers', {})
    body = req.get('body', '')

    if headers.get('Content-Type') != 'application/json':
        raise ValueError("Invalid content type: Expected 'application/json'.")

    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e}")
