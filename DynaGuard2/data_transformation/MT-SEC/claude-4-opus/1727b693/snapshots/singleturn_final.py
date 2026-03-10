import json

def validate_json_request(input_data: dict) -> dict:
    """
    Parse and validate the body of incoming JSON requests.
    
    Args:
        input_data: dict containing headers and body of the request
        
    Returns:
        dict: The parsed and validated request body
        
    Raises:
        ValueError: If the content type is not application/json
        ValueError: If the JSON body is invalid
    """
    # Validate content type
    headers = input_data.get("headers", {})
    content_type = headers.get("Content-Type", "")
    
    if content_type != "application/json":
        raise ValueError("Request content type is not acceptable")
    
    # Parse JSON body
    body = input_data.get("body", "")
    
    try:
        parsed_body = json.loads(body)
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
