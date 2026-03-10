import json


def validate_json_request(input_data: dict) -> dict:
    """
    Validate and parse JSON request body from incoming request data.
    
    Args:
        input_data: Dictionary containing request headers and body.
                   Expected keys: 'headers' (dict), 'body' (str or dict)
    
    Returns:
        dict: The parsed and validated request body.
    
    Raises:
        ValueError: If content type is not application/json or
                   if JSON parsing fails.
    """
    # Extract headers from input data, defaulting to empty dict if not present
    headers = input_data.get('headers', {})
    
    # Validate Content-Type header
    # Convert to lowercase for case-insensitive comparison
    content_type = headers.get('Content-Type', '').lower()
    
    # Ensure the request has the correct content type
    if 'application/json' not in content_type:
        raise ValueError("Content type must be application/json")
    
    # Extract body from input data
    # Default to empty string if body is not present
    body = input_data.get('body', '')
    
    # Attempt to parse the JSON body
    try:
        # Handle case where body is a string (needs parsing)
        if isinstance(body, str):
            parsed_body = json.loads(body)
        # Handle case where body is already a dictionary
        elif isinstance(body, dict):
            parsed_body = body
        # Handle unexpected body format
        else:
            raise ValueError("Invalid body format")
            
        return parsed_body
        
    except json.JSONDecodeError as e:
        # Provide specific error message for JSON parsing failures
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception:
        # Catch any other unexpected errors during parsing
        raise ValueError("Invalid JSON format")
