import json
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def validate_json_request(input_data: dict) -> dict:
    """
    Validate and parse JSON request body from incoming request data.
    
    Args:
        input_data: Dictionary containing request headers and body.
                   Expected keys: 'headers' (dict), 'body' (str or dict)
    
    Returns:
        dict: The parsed and validated request body.
    
    Raises:
        ValueError: If content type is not application/json or
                   if JSON parsing fails.
    """
    # Extract headers from input data, defaulting to empty dict if not present
    headers = input_data.get('headers', {})
    
    # Validate Content-Type header
    # Convert to lowercase for case-insensitive comparison
    content_type = headers.get('Content-Type', '').lower()
    
    # Guard clause: Ensure the request has the correct content type
    if 'application/json' not in content_type:
        logger.error(f"Invalid content type: {content_type}")
        raise ValueError("Content type must be application/json")
    
    # Extract body from input data
    # Default to empty string if body is not present
    body = input_data.get('body', '')
    
    # Guard clause: Handle already parsed dictionary
    if isinstance(body, dict):
        logger.debug("Body is already a dictionary, returning as-is")
        return body
    
    # Guard clause: Ensure body is a string for JSON parsing
    if not isinstance(body, str):
        logger.error(f"Invalid body type: {type(body)}")
        raise ValueError("Invalid body format")
    
    # Attempt to parse the JSON string
    try:
        parsed_body = json.loads(body)
        logger.debug("Successfully parsed JSON body")
        return parsed_body
        
    except json.JSONDecodeError as e:
        # Log and provide specific error message for JSON parsing failures
        logger.error(f"JSON decoding failed: {str(e)}")
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        # Log and catch any other unexpected errors during parsing
        logger.error(f"Unexpected error during JSON parsing: {str(e)}")
        raise ValueError("Invalid JSON format")
