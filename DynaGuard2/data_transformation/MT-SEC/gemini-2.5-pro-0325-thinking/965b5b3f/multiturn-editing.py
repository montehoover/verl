import json

def process_json_request(req_data):
    """
    Processes and validates the body of incoming JSON requests.

    Checks if the content type is 'application/json', decodes the JSON body,
    and returns it as a structured dictionary.

    Args:
        req_data (dict): A dictionary containing 'headers' and 'body'.
                         Example: {'headers': {'Content-Type': 'application/json'}, 'body': '{"key": "value"}'}

    Returns:
        dict: The decoded JSON body.

    Raises:
        ValueError: If the content type is not 'application/json' or if there are JSON format issues.
    """
    headers = req_data.get('headers', {})
    body = req_data.get('body', '')

    content_type = headers.get('Content-Type')
    if content_type != 'application/json':
        raise ValueError("Invalid content type. Only 'application/json' is accepted.")

    try:
        if not body: # Handle empty body case before attempting to parse
            raise ValueError("Request body is empty. JSON data expected.")
        decoded_body = json.loads(body)
        return decoded_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e: # Catch any other unexpected errors during processing
        raise ValueError(f"Error processing JSON request: {e}")
