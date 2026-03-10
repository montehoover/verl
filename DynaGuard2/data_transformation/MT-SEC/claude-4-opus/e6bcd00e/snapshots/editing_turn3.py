import json

def analyze_json_request(incoming_request):
    """
    Parses JSON request bodies in FastAPI.
    
    Args:
        incoming_request (dict): Dictionary containing HTTP headers and body
        
    Returns:
        dict: Parsed request body as a dictionary
        
    Raises:
        ValueError: When content type is incorrect or JSON decoding fails
    """
    headers = incoming_request.get('headers', {})
    content_type = headers.get('Content-Type', '')
    
    if content_type.lower() != 'application/json':
        raise ValueError(f"Invalid content type: {content_type}. Expected 'application/json'")
    
    body = incoming_request.get('body', '')
    
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {str(e)}")
