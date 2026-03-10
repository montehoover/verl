import json

def validate_content_type(headers):
    """
    Validates if the Content-Type header is application/json.
    
    Args:
        headers (dict): Dictionary representing request headers
        
    Returns:
        bool: True if Content-Type is application/json, False otherwise
    """
    content_type = headers.get('Content-Type', '').lower()
    return content_type == 'application/json'

def extract_json_body(request_data):
    """
    Extracts and decodes JSON data from the request body.
    
    Args:
        request_data (dict): Dictionary containing a 'body' key with JSON string
        
    Returns:
        dict: Decoded JSON data
        
    Raises:
        ValueError: If JSON decoding fails or 'body' key is missing
    """
    if 'body' not in request_data:
        raise ValueError("Request data must contain a 'body' key")
    
    body = request_data['body']
    
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in request body: {str(e)}")
