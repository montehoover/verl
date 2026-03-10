import json

def process_json_request(req_data: dict) -> dict:
    # Check content type
    headers = req_data.get('headers', {})
    content_type = headers.get('Content-Type', '')
    
    if not content_type.startswith('application/json'):
        raise ValueError("Content type must be application/json")
    
    # Get and validate body
    body = req_data.get('body', '')
    
    if not body:
        raise ValueError("Request body is empty")
    
    try:
        # Parse JSON body
        parsed_body = json.loads(body)
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception:
        raise ValueError("Failed to process request body")
