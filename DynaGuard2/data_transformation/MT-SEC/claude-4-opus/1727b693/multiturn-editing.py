import json

def validate_json_request(input_data):
    headers = input_data.get('headers', {})
    content_type = headers.get('Content-Type', '')
    
    if content_type != 'application/json':
        raise ValueError("Content-Type must be 'application/json'")
    
    body = input_data.get('body', '')
    
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in request body")
