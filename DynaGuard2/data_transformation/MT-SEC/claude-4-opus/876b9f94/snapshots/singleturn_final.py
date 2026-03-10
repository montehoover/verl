import json

def process_json_input(req: dict) -> dict:
    """
    Parse JSON request bodies in FastAPI.
    
    Args:
        req: A dictionary containing both the HTTP headers and body of the incoming request
        
    Returns:
        The parsed request body in dictionary format if valid
        
    Raises:
        ValueError: When content type is incorrect or JSON decoding fails
    """
    # Check if headers exist and get content type
    headers = req.get("headers", {})
    content_type = headers.get("Content-Type", "")
    
    # Validate content type
    if content_type != "application/json":
        raise ValueError(f"Invalid content type: {content_type}. Expected application/json")
    
    # Get the body from the request
    body = req.get("body", "")
    
    # Parse the JSON body
    try:
        parsed_json = json.loads(body)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {str(e)}")
