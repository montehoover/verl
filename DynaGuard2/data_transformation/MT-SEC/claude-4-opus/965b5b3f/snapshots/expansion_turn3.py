import json

def validate_content_type(headers: dict) -> bool:
    """
    Validates if the Content-Type header is 'application/json'.
    
    Args:
        headers: Dictionary representing request headers
        
    Returns:
        bool: True if Content-Type is 'application/json', False otherwise
    """
    content_type = headers.get('Content-Type', '')
    return content_type.lower() == 'application/json'


def decode_json_body(body: str) -> dict:
    """
    Decodes JSON data from the request body.
    
    Args:
        body: String representing the JSON body
        
    Returns:
        dict: The decoded JSON data as a dictionary
        
    Raises:
        ValueError: If the JSON is malformed
    """
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def process_json_request(req_data: dict) -> dict:
    """
    Processes incoming JSON requests by validating content type and decoding the body.
    
    Args:
        req_data: Dictionary containing 'headers' and 'body'
        
    Returns:
        dict: The parsed JSON data
        
    Raises:
        ValueError: If content type is incorrect or JSON is malformed
    """
    headers = req_data.get('headers', {})
    body = req_data.get('body', '')
    
    if not validate_content_type(headers):
        raise ValueError("Invalid content type: Expected 'application/json'")
    
    return decode_json_body(body)
