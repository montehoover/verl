import json

def process_json_request(req_data):
    """
    Processes and validates the body of incoming JSON requests.
    
    Args:
        req_data (dict): A dictionary including headers and body.
        
    Returns:
        dict: The decoded JSON body as a structured dictionary.
        
    Raises:
        ValueError: If content type is not 'application/json' or if JSON decoding fails.
    """
    headers = req_data.get('headers', {})
    content_type = headers.get('Content-Type', '')
    
    if content_type.lower() != 'application/json':
        raise ValueError("Content type must be 'application/json'")
    
    body = req_data.get('body', '')
    
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
