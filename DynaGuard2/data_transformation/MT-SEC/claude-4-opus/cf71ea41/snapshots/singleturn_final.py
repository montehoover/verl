import json

def process_json_payload(req_data: dict) -> dict:
    """
    Process and validate incoming JSON requests in FastAPI.
    
    Args:
        req_data: Dictionary that includes both the headers and body of the HTTP request
        
    Returns:
        dict: Contains the validated and parsed request body
        
    Raises:
        ValueError: If the JSON in the request body is malformed
        ValueError: If the content-type is not set to a JSON-compatible format
    """
    # Check if headers exist in req_data
    headers = req_data.get('headers', {})
    
    # Validate content-type header
    content_type = headers.get('Content-Type', '').lower()
    
    # Check if content-type is JSON-compatible
    json_compatible_types = ['application/json', 'application/json; charset=utf-8']
    if not any(content_type.startswith(jtype) for jtype in json_compatible_types):
        raise ValueError("Content-Type header must be set to a JSON-compatible format")
    
    # Get the body from req_data
    body = req_data.get('body', '')
    
    # Parse the JSON body
    try:
        parsed_body = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in request body: {str(e)}")
    except TypeError as e:
        raise ValueError(f"Invalid request body type: {str(e)}")
    
    return parsed_body
