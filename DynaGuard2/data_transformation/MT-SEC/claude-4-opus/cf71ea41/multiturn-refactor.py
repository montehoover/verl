import json
import logging


def process_json_payload(req_data: dict) -> dict:
    """
    Process and validate JSON payload from HTTP request data.
    
    This function extracts the JSON body from a request dictionary containing
    headers and body, validates that the content-type is JSON-compatible,
    and parses the JSON content.
    
    Args:
        req_data: Dictionary containing HTTP request data with keys:
            - 'headers': Dict with HTTP headers (must include 'content-type')
            - 'body': String containing the request body
    
    Returns:
        Dictionary containing the parsed JSON content from the request body.
    
    Raises:
        ValueError: If content-type header is missing or not JSON-compatible.
        ValueError: If request body is empty or contains malformed JSON.
    """
    # Initialize logger for this function
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Create console handler if logger has no handlers
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.debug("Starting JSON payload processing")
    
    # Extract headers from request data
    headers = req_data.get('headers', {})
    content_type = headers.get('content-type', '').lower()
    
    logger.debug(f"Request headers: {headers}")
    logger.debug(f"Content-Type: {content_type}")
    
    # Validate content-type header exists
    if not content_type:
        logger.error("Content-Type header is missing")
        raise ValueError("Content-Type header is missing")
    
    # Validate content-type is JSON-compatible
    valid_json_types = ['application/json', 'text/json']
    if not any(json_type in content_type for json_type in valid_json_types):
        logger.error(f"Invalid Content-Type: {content_type}")
        raise ValueError(
            f"Content-Type must be application/json or text/json, got: {content_type}"
        )
    
    # Extract body from request data
    body = req_data.get('body', '')
    
    logger.debug(f"Request body length: {len(body)} characters")
    
    # Validate body is not empty
    if not body:
        logger.error("Request body is empty")
        raise ValueError("Request body is empty")
    
    # Parse JSON content from body
    try:
        parsed_body = json.loads(body)
        logger.info(f"Successfully parsed JSON payload with {len(parsed_body)} keys" if isinstance(parsed_body, dict) else "Successfully parsed JSON payload")
        logger.debug(f"Parsed content: {parsed_body}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {str(e)}")
        logger.debug(f"Malformed body content: {body[:100]}..." if len(body) > 100 else f"Malformed body content: {body}")
        raise ValueError(f"Malformed JSON in request body: {str(e)}")
    
    logger.debug("JSON payload processing completed successfully")
    return parsed_body
