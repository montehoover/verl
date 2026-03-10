import json

def validate_json_request(input_data: dict) -> dict:
    # Check if Content-Type header exists and is application/json
    headers = input_data.get('headers', {})
    content_type = headers.get('Content-Type', '').lower()
    
    if 'application/json' not in content_type:
        raise ValueError("Content type must be application/json")
    
    # Get the body from input_data
    body = input_data.get('body', '')
    
    # Try to parse the JSON body
    try:
        if isinstance(body, str):
            parsed_body = json.loads(body)
        elif isinstance(body, dict):
            # Body might already be parsed
            parsed_body = body
        else:
            raise ValueError("Invalid body format")
            
        return parsed_body
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception:
        raise ValueError("Invalid JSON format")
