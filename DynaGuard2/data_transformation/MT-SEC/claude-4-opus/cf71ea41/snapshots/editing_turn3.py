import json

def process_json_payload(req_data):
    """
    Handles the parsing and validation of incoming JSON requests in a FastAPI application.
    
    Args:
        req_data (dict): A dictionary including both headers and body of the HTTP request
        
    Returns:
        dict: The validated and parsed request body as a dictionary
        
    Raises:
        ValueError: If the JSON in the request body is malformed or if the content-type 
                    is not set to a JSON-compatible format
    """
    # Check if headers exist in req_data
    headers = req_data.get('headers', {})
    
    # Check Content-Type
    content_type = headers.get('Content-Type', '').lower()
    if not ('application/json' in content_type or 'application/ld+json' in content_type):
        raise ValueError("Content-Type must be a JSON-compatible format")
    
    # Get body from req_data
    body = req_data.get('body', '')
    
    # Parse JSON
    try:
        if isinstance(body, str):
            parsed_body = json.loads(body)
        elif isinstance(body, dict):
            parsed_body = body
        else:
            raise ValueError("Request body must be a string or dictionary")
            
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in request body: {str(e)}")
