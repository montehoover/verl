import json

def process_json_input(req):
    """
    Parses JSON request bodies in FastAPI.
    
    Args:
        req (dict): Dictionary containing both HTTP headers and body
        
    Returns:
        dict: Parsed request body as a dictionary
        
    Raises:
        ValueError: When content type is incorrect or JSON decoding fails
    """
    # Check content type
    headers = req.get('headers', {})
    content_type = headers.get('Content-Type', '')
    if content_type.lower() != 'application/json':
        raise ValueError("Content-Type must be 'application/json'")
    
    # Parse JSON body
    body = req.get('body', '')
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")
