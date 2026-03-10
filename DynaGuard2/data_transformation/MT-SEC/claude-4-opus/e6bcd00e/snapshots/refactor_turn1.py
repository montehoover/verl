import json

def analyze_json_request(incoming_request: dict) -> dict:
    # Check if the request has headers
    if 'headers' not in incoming_request:
        raise ValueError("Request must contain headers")
    
    headers = incoming_request['headers']
    
    # Check content-type header
    content_type = None
    for key, value in headers.items():
        if key.lower() == 'content-type':
            content_type = value
            break
    
    if not content_type:
        raise ValueError("Content-Type header is missing")
    
    # Verify content type is JSON
    if 'application/json' not in content_type.lower():
        raise ValueError(f"Invalid content type: {content_type}. Expected application/json")
    
    # Check if body exists
    if 'body' not in incoming_request:
        raise ValueError("Request body is missing")
    
    body = incoming_request['body']
    
    # Parse JSON body
    try:
        if isinstance(body, str):
            parsed_body = json.loads(body)
        elif isinstance(body, dict):
            # Body is already a dictionary
            parsed_body = body
        else:
            raise ValueError("Request body must be a string or dictionary")
            
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in request body: {str(e)}")
