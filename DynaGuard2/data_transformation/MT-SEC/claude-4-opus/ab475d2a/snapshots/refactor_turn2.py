import json


def validate_content_type(headers: dict) -> None:
    """Validate that the request has JSON content type.
    
    Args:
        headers: Dictionary of request headers
        
    Raises:
        ValueError: If content type is not application/json
    """
    content_type = headers.get('content-type', '').lower()
    
    if 'application/json' not in content_type:
        raise ValueError("Request content type is not acceptable")


def parse_json_body(body: str) -> dict:
    """Parse JSON string into a dictionary.
    
    Args:
        body: JSON string to parse
        
    Returns:
        Parsed dictionary from JSON
        
    Raises:
        ValueError: If body is empty, invalid JSON, or not an object
    """
    if not body:
        raise ValueError("Request body is empty")
    
    try:
        parsed_body = json.loads(body)
        
        if not isinstance(parsed_body, dict):
            raise ValueError("JSON body must be an object")
            
        return parsed_body
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception:
        raise ValueError("Failed to decode JSON request body")


def decode_json_request(req: dict) -> dict:
    """Parse and validate JSON request body.
    
    Args:
        req: Request object with headers and body
        
    Returns:
        Parsed request body as dictionary
        
    Raises:
        ValueError: If content type or JSON format is invalid
    """
    headers = req.get('headers', {})
    body = req.get('body', '')
    
    validate_content_type(headers)
    return parse_json_body(body)
