import json

def decode_json_request(req: dict) -> dict:
    # Check content type
    headers = req.get('headers', {})
    content_type = headers.get('content-type', '').lower()
    
    if 'application/json' not in content_type:
        raise ValueError("Request content type is not acceptable")
    
    # Get and parse body
    body = req.get('body', '')
    
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
