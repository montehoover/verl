import json

def process_json_input(req: dict) -> dict:
    # Check if 'headers' exists in the request
    headers = req.get('headers', {})
    
    # Check content type
    content_type = headers.get('content-type', '').lower()
    if 'application/json' not in content_type:
        raise ValueError(f"Incorrect content type: expected 'application/json', got '{content_type}'")
    
    # Get the body from the request
    body = req.get('body', '')
    
    # Try to parse the JSON
    try:
        parsed_json = json.loads(body)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {str(e)}")
