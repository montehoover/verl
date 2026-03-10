import json

def process_json_payload(req_data: dict) -> dict:
    # Check if headers exist and content-type is JSON-compatible
    headers = req_data.get('headers', {})
    content_type = headers.get('content-type', '').lower()
    
    if not content_type:
        raise ValueError("Content-Type header is missing")
    
    # Check for JSON-compatible content types
    if 'application/json' not in content_type and 'text/json' not in content_type:
        raise ValueError(f"Content-Type must be application/json or text/json, got: {content_type}")
    
    # Get the body from request data
    body = req_data.get('body', '')
    
    if not body:
        raise ValueError("Request body is empty")
    
    # Parse the JSON body
    try:
        parsed_body = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in request body: {str(e)}")
    
    return parsed_body
