import json
import logging


# Configure logger
logger = logging.getLogger(__name__)


def validate_content_type(headers: dict) -> None:
    """Validate that the request has JSON content type.
    
    Args:
        headers: Dictionary of request headers
        
    Raises:
        ValueError: If content type is not application/json
    """
    content_type = headers.get('content-type', '').lower()
    logger.debug(f"Validating content type: {content_type}")
    
    if 'application/json' not in content_type:
        logger.error(f"Invalid content type: {content_type}")
        raise ValueError("Request content type is not acceptable")
    
    logger.debug("Content type validation successful")


def parse_json_body(body: str) -> dict:
    """Parse JSON string into a dictionary.
    
    Args:
        body: JSON string to parse
        
    Returns:
        Parsed dictionary from JSON
        
    Raises:
        ValueError: If body is empty, invalid JSON, or not an object
    """
    if not body:
        logger.error("Empty request body received")
        raise ValueError("Request body is empty")
    
    logger.debug(f"Parsing JSON body of length: {len(body)}")
    
    try:
        parsed_body = json.loads(body)
        
        if not isinstance(parsed_body, dict):
            logger.error(f"JSON body is not an object, got type: {type(parsed_body).__name__}")
            raise ValueError("JSON body must be an object")
        
        logger.debug(f"Successfully parsed JSON with {len(parsed_body)} keys")
        return parsed_body
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error at position {e.pos}: {str(e)}")
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during JSON parsing: {type(e).__name__}: {str(e)}")
        raise ValueError("Failed to decode JSON request body")


def decode_json_request(req: dict) -> dict:
    """Parse and validate JSON request body.
    
    Args:
        req: Request object with headers and body
        
    Returns:
        Parsed request body as dictionary
        
    Raises:
        ValueError: If content type or JSON format is invalid
    """
    logger.info("Starting JSON request decoding")
    
    headers = req.get('headers', {})
    body = req.get('body', '')
    
    logger.debug(f"Request headers: {headers}")
    
    try:
        validate_content_type(headers)
        parsed_data = parse_json_body(body)
        
        logger.info(f"Successfully decoded JSON request with {len(parsed_data)} fields")
        return parsed_data
        
    except ValueError as e:
        logger.error(f"Request validation failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in decode_json_request: {type(e).__name__}: {str(e)}")
        raise ValueError(f"Failed to process request: {str(e)}")
