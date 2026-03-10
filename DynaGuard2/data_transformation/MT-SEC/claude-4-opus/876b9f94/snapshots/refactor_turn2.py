import json


def process_json_input(req: dict) -> dict:
    """
    Parse JSON request bodies in FastAPI.
    
    This function ensures the JSON content is properly structured and valid,
    returning the JSON object as a dictionary. It validates that the request
    has the correct content type before attempting to parse the body.
    
    Args:
        req (dict): A dictionary containing both the HTTP headers and body 
                    of the incoming request. Expected structure:
                    {
                        'headers': {'content-type': 'application/json', ...},
                        'body': '{"key": "value", ...}'
                    }
    
    Returns:
        dict: The parsed request body in dictionary format if valid.
    
    Raises:
        ValueError: When the content type is incorrect or when the JSON 
                    decoding process fails due to malformed JSON.
    """
    # Extract headers from the request, defaulting to empty dict if not present
    headers = req.get('headers', {})
    
    # Validate content type
    content_type = headers.get('content-type', '').lower()
    if 'application/json' not in content_type:
        raise ValueError(
            f"Incorrect content type: expected 'application/json', "
            f"got '{content_type}'"
        )
    
    # Extract the request body
    body = req.get('body', '')
    
    # Attempt to parse the JSON body
    try:
        parsed_json = json.loads(body)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {str(e)}")
