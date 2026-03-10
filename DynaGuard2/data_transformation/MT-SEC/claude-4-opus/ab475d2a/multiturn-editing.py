import json

def decode_json_request(req):
    """
    Processes incoming JSON requests in a FastAPI application.
    
    Args:
        req (dict): A dictionary containing both headers and body.
        
    Returns:
        dict: The decoded JSON body as a structured dictionary.
        
    Raises:
        ValueError: If content type is not 'application/json' or if JSON decoding fails.
    """
    headers = req.get('headers', {})
    content_type = headers.get('Content-Type', '')
    
    if content_type.lower() != 'application/json':
        raise ValueError("Invalid content type")
    
    body = req.get('body', '')
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")
