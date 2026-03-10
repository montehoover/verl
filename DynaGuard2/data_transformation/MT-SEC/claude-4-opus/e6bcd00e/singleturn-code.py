import json

def analyze_json_request(incoming_request: dict) -> dict:
    """
    Parse JSON request bodies in FastAPI.
    
    Args:
        incoming_request: A dictionary containing both the HTTP headers and body of the incoming request.
        
    Returns:
        The parsed request body in dictionary format if valid.
        
    Raises:
        ValueError: When the content type is incorrect or when the JSON decoding process fails.
    """
    # Check if headers exist and get content type
    headers = incoming_request.get("headers", {})
    content_type = headers.get("Content-Type", "")
    
    # Validate content type
    if content_type != "application/json":
        raise ValueError(f"Invalid content type: expected 'application/json', got '{content_type}'")
    
    # Get the body
    body = incoming_request.get("body", "")
    
    # Parse JSON
    try:
        parsed_json = json.loads(body)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {str(e)}")
