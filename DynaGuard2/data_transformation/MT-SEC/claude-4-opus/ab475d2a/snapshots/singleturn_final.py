import json

def decode_json_request(req: dict) -> dict:
    """
    Parse and validate the body of incoming JSON requests.
    
    Args:
        req: dict - Request object containing headers and body
        
    Returns:
        dict - The parsed and validated request body
        
    Raises:
        ValueError: If content type is not application/json
        ValueError: If JSON parsing fails
    """
    # Check content type
    headers = req.get("headers", {})
    content_type = headers.get("Content-Type", "")
    
    if content_type != "application/json":
        raise ValueError("Content type must be application/json")
    
    # Parse JSON body
    try:
        body = req.get("body", "")
        parsed_data = json.loads(body)
        return parsed_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing request body: {str(e)}")
