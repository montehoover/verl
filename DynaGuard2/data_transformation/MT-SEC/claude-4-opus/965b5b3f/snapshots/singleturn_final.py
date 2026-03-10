import json

def process_json_request(req_data: dict) -> dict:
    """
    Parse and validate the body of incoming JSON requests.
    
    Args:
        req_data: dict containing headers and body of the request
        
    Returns:
        dict: The parsed and validated request body
        
    Raises:
        ValueError: If content type is not application/json
        ValueError: If JSON parsing fails
    """
    # Check if headers exist and validate content type
    headers = req_data.get("headers", {})
    content_type = headers.get("Content-Type", "")
    
    if content_type != "application/json":
        raise ValueError("Request content type is not acceptable")
    
    # Get the body from request data
    body = req_data.get("body", "")
    
    # Parse JSON body
    try:
        parsed_body = json.loads(body)
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
