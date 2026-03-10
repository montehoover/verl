import json


def analyze_json_request(incoming_request: dict) -> dict:
    """
    Parse and validate JSON request bodies from incoming HTTP requests.
    
    This function processes incoming requests in FastAPI applications by validating
    the content type and parsing the JSON body. It ensures the request contains
    proper headers and valid JSON content before returning the parsed data.
    
    Args:
        incoming_request (dict): A dictionary containing both the HTTP headers 
                                and body of the incoming request. Expected structure:
                                {
                                    'headers': {'content-type': 'application/json', ...},
                                    'body': '{"key": "value"}' or {"key": "value"}
                                }
    
    Returns:
        dict: The parsed request body in dictionary format if valid.
    
    Raises:
        ValueError: When the content type is incorrect, headers/body are missing,
                   or when the JSON decoding process fails due to malformed JSON.
    """
    # Validate that headers are present in the request
    if 'headers' not in incoming_request:
        raise ValueError("Request must contain headers")
    
    headers = incoming_request['headers']
    
    # Extract content-type header (case-insensitive search)
    content_type = None
    for key, value in headers.items():
        if key.lower() == 'content-type':
            content_type = value
            break
    
    # Ensure content-type header exists
    if not content_type:
        raise ValueError("Content-Type header is missing")
    
    # Validate that content type is JSON
    if 'application/json' not in content_type.lower():
        raise ValueError(f"Invalid content type: {content_type}. Expected application/json")
    
    # Validate that request body exists
    if 'body' not in incoming_request:
        raise ValueError("Request body is missing")
    
    body = incoming_request['body']
    
    # Parse the JSON body based on its type
    try:
        if isinstance(body, str):
            # Parse string JSON into dictionary
            parsed_body = json.loads(body)
        elif isinstance(body, dict):
            # Body is already a dictionary, no parsing needed
            parsed_body = body
        else:
            # Body is neither string nor dictionary
            raise ValueError("Request body must be a string or dictionary")
            
        return parsed_body
        
    except json.JSONDecodeError as e:
        # Re-raise JSON decoding errors with a more descriptive message
        raise ValueError(f"Invalid JSON in request body: {str(e)}")
