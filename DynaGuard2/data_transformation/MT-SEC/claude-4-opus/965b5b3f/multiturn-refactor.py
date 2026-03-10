import json
import logging

# Configure logger
logger = logging.getLogger(__name__)

def validate_content_type(headers: dict) -> None:
    """Validate that the content type is application/json."""
    content_type = headers.get('Content-Type', '')
    logger.debug(f"Validating content type: {content_type}")
    
    if not content_type.startswith('application/json'):
        logger.error(f"Invalid content type: {content_type}. Expected application/json")
        raise ValueError("Content type must be application/json")
    
    logger.debug("Content type validation successful")

def decode_json_body(body: str) -> dict:
    """Decode JSON body string into a dictionary."""
    if not body:
        logger.error("Request body is empty")
        raise ValueError("Request body is empty")
    
    logger.debug(f"Attempting to decode JSON body of length: {len(body)}")
    
    try:
        result = json.loads(body)
        logger.debug(f"Successfully decoded JSON body with {len(result)} keys")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error at line {e.lineno}, column {e.colno}: {e.msg}")
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during JSON decoding: {type(e).__name__}: {str(e)}")
        raise ValueError("Failed to process request body")

def process_json_request(req_data: dict) -> dict:
    logger.info("Starting JSON request processing")
    
    try:
        # Validate content type
        headers = req_data.get('headers', {})
        logger.debug(f"Request headers: {headers}")
        validate_content_type(headers)
        
        # Decode JSON body
        body = req_data.get('body', '')
        result = decode_json_body(body)
        
        logger.info("JSON request processing completed successfully")
        return result
    except ValueError as e:
        logger.error(f"Request processing failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during request processing: {type(e).__name__}: {str(e)}")
        raise
