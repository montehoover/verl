import json

def validate_content_type(headers: dict) -> None:
    """Validate that the content type is application/json."""
    content_type = headers.get('Content-Type', '')
    if not content_type.startswith('application/json'):
        raise ValueError("Content type must be application/json")

def decode_json_body(body: str) -> dict:
    """Decode JSON body string into a dictionary."""
    if not body:
        raise ValueError("Request body is empty")
    
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception:
        raise ValueError("Failed to process request body")

def process_json_request(req_data: dict) -> dict:
    # Validate content type
    headers = req_data.get('headers', {})
    validate_content_type(headers)
    
    # Decode JSON body
    body = req_data.get('body', '')
    return decode_json_body(body)
