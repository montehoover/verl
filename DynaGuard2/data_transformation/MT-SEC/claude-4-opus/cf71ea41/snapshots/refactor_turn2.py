import json


def process_json_payload(req_data: dict) -> dict:
    """
    Process and validate JSON payload from HTTP request data.
    
    This function extracts the JSON body from a request dictionary containing
    headers and body, validates that the content-type is JSON-compatible,
    and parses the JSON content.
    
    Args:
        req_data: Dictionary containing HTTP request data with keys:
            - 'headers': Dict with HTTP headers (must include 'content-type')
            - 'body': String containing the request body
    
    Returns:
        Dictionary containing the parsed JSON content from the request body.
    
    Raises:
        ValueError: If content-type header is missing or not JSON-compatible.
        ValueError: If request body is empty or contains malformed JSON.
    """
    # Extract headers from request data
    headers = req_data.get('headers', {})
    content_type = headers.get('content-type', '').lower()
    
    # Validate content-type header exists
    if not content_type:
        raise ValueError("Content-Type header is missing")
    
    # Validate content-type is JSON-compatible
    valid_json_types = ['application/json', 'text/json']
    if not any(json_type in content_type for json_type in valid_json_types):
        raise ValueError(
            f"Content-Type must be application/json or text/json, got: {content_type}"
        )
    
    # Extract body from request data
    body = req_data.get('body', '')
    
    # Validate body is not empty
    if not body:
        raise ValueError("Request body is empty")
    
    # Parse JSON content from body
    try:
        parsed_body = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in request body: {str(e)}")
    
    return parsed_body
